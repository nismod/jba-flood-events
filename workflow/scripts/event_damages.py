"""Depth and damage for every asset piece and event in one HAZ.

Inputs are the piece table from `split_assets.py` and the (HAZ, event) return
periods from `haz_event_rp.py`. Depths are interpolated from the mapped RP
depths of each piece's cell (see `floodlib.depth`). The defended case applies
"zero depth if the event RP is at or below the piece's standard of protection".

The output has one row per (piece, event) with depth > 0:

    split_id, id, event_id, rp, depth_m, protected, damage_fraction,
    damage_undefended_usd, damage_defended_usd

Damage is null for exposure-only pieces, i.e. those with no cost_type.

Example:

    python workflow/scripts/event_damages.py \
        --haz_id 500_13_19495 \
        --splits processed_data/exposure/road_edges/haz=500_13_19495/splits.parquet \
        --haz_event_rp processed_data/prepared/events/Sim_haz_event_rp.parquet \
        --rp 20 --rp 50 --rp 100 --rp 200 --rp 500 --rp 1500 --horp 11 \
        --layer road_edges --cost_table processed_data/costs/road_edges.csv \
        --curves_dir processed_data/curves/road \
        --output processed_data/event_damages/Sim/road_edges/haz=500_13_19495/part.parquet
"""

import logging
import os

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from floodlib.damage import check_costs_and_curves, damage_fraction, load_curves, load_unit_costs
from floodlib.depth import event_depths, zero_depth_rp
from floodlib.events import EVENT_ID, HAZ_ID
from floodlib.split import layer_kind

# Cap the size of the (pieces x events) depth array computed at once
MAX_CELLS_PER_CHUNK = 5_000_000


def output_schema(event_id_type: pa.DataType, id_type: pa.DataType) -> pa.Schema:
    return pa.schema(
        [
            ("split_id", pa.int32()),
            ("id", id_type),
            ("event_id", event_id_type),
            ("rp", pa.float32()),
            ("depth_m", pa.float32()),
            ("protected", pa.bool_()),
            ("damage_fraction", pa.float32()),
            ("damage_undefended_usd", pa.float64()),
            ("damage_defended_usd", pa.float64()),
        ]
    )


@click.command()
@click.option("--haz_id", required=True)
@click.option("--splits", "splits_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--haz_event_rp", "rp_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--rp", "rps", required=True, multiple=True, type=float)
@click.option("--horp", required=True, type=float, help="Hazard onset return period")
@click.option("--layer", required=True)
@click.option("--cost_table", default=None, type=click.Path(dir_okay=False))
@click.option("--curves_dir", default=None, type=click.Path(file_okay=False))
@click.option("--output", required=True, type=click.Path(dir_okay=False, writable=True))
def main(haz_id, splits_path, rp_path, rps, horp, layer, cost_table, curves_dir, output):
    rps = np.array(sorted(rps), dtype=float)
    depth_cols = [f"depth_rp{rp:g}" for rp in rps]
    kind = layer_kind(layer)

    splits = pd.read_parquet(
        splits_path, columns=["split_id", "id", "cost_type", "size", "sop", *depth_cols]
    )
    events = pd.read_parquet(rp_path, filters=[(HAZ_ID, "==", str(haz_id))], columns=[EVENT_ID, "rp"])
    logging.info("%d pieces, %d events for HAZ %s", len(splits), len(events), haz_id)

    depths = splits[depth_cols].to_numpy(dtype=np.float64)
    wet = (depths > 0).any(axis=1)
    splits, depths = splits[wet].reset_index(drop=True), depths[wet]

    schema = output_schema(
        pq.read_schema(rp_path).field(EVENT_ID).type,
        pq.read_schema(splits_path).field("id").type,
    )

    unit_cost, curves = costing(splits, kind, layer, cost_table, curves_dir)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    n_rows = 0
    with pq.ParquetWriter(output, schema) as writer:
        if len(splits) and len(events):
            first_wet, z = zero_depth_rp(depths, rps, horp)
            sop = splits["sop"].to_numpy(dtype=float)
            chunk = max(1, MAX_CELLS_PER_CHUNK // len(splits))
            for start in range(0, len(events), chunk):
                ev = events.iloc[start : start + chunk]
                table = chunk_rows(splits, depths, first_wet, z, sop, unit_cost, curves, ev, rps)
                writer.write_table(pa.Table.from_pandas(table, schema=schema, preserve_index=False))
                n_rows += len(table)
        else:
            writer.write_table(schema.empty_table())
    logging.info("Wrote %d (piece, event) rows to %s", n_rows, output)


def costing(splits, kind, layer, cost_table, curves_dir):
    """Unit cost per piece (NaN if exposure-only) and curves by cost_type."""
    costed = splits["cost_type"].notna()
    if not costed.any():
        return np.full(len(splits), np.nan), {}
    if cost_table is None or not os.path.exists(cost_table):
        raise ValueError(f"{layer}: pieces have a cost_type but there is no cost table ({cost_table})")
    unit_costs = load_unit_costs(cost_table, kind)
    curves = load_curves(curves_dir) if curves_dir and os.path.isdir(curves_dir) else {}
    check_costs_and_curves(splits["cost_type"], unit_costs, curves, layer)
    unit_cost = splits["cost_type"].map(unit_costs).to_numpy(dtype=float)
    return unit_cost, curves


def chunk_rows(splits, depths, first_wet, z, sop, unit_cost, curves, events, rps) -> pd.DataFrame:
    event_rp = events["rp"].to_numpy(dtype=float)
    d = event_depths(depths, rps, first_wet, z, event_rp)
    p, e = np.nonzero(d > 0)
    depth = d[p, e]
    rp = event_rp[e]
    frac = damage_fraction(splits["cost_type"].to_numpy()[p], depth, curves)
    damage = splits["size"].to_numpy()[p] * unit_cost[p] * frac
    protected = rp <= np.nan_to_num(sop[p], nan=-np.inf)
    return pd.DataFrame(
        {
            "split_id": splits["split_id"].to_numpy()[p],
            "id": splits["id"].to_numpy()[p],
            "event_id": events[EVENT_ID].to_numpy()[e],
            "rp": rp.astype(np.float32),
            "depth_m": depth.astype(np.float32),
            "protected": protected,
            "damage_fraction": frac.astype(np.float32),
            "damage_undefended_usd": damage,
            "damage_defended_usd": np.where(protected & ~np.isnan(damage), 0.0, damage),
        }
    )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
