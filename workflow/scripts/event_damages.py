"""Depth and damage per asset and event, for every layer and event set in one HAZ.

Inputs are the piece tables from `split_assets.py` and the (HAZ, event) return
periods from `haz_event_rp.py`. Depths are interpolated from the mapped RP
depths of each piece's cell (see `jba_events.depth`), then pieces are summed
to one row per (asset, adm1, event); see `jba_events.damage.asset_event_chunks`.

The piece-by-event table isn't stored. Rebuild it when needed with
`jba_events.damage.piece_event_table`.

Example:

    python workflow/scripts/event_damages.py \
        --haz_id 500_13_19495 --layer road_edges --layer railway_edges --type Obs --type Sim \
        --splits_template 'processed_data/exposure/{layer}/haz=500_13_19495/splits.parquet' \
        --haz_event_rp_template 'processed_data/prepared/events/{TYPE}_haz_event_rp.parquet' \
        --rp 20 --rp 50 --rp 100 --rp 200 --rp 500 --rp 1500 --horp 11 \
        --cost_template 'processed_data/costs/{layer}.csv' \
        --curves_template 'processed_data/curves/{asset_type}' \
        --output_template 'processed_data/asset_event_damages/{TYPE}/{layer}/haz=500_13_19495/part.parquet'
"""

import logging
import os

import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from jba_events.damage import PieceCosting, asset_event_chunks
from jba_events.events import EVENT_ID, HAZ_ID
from jba_events.split import layer_asset_type, layer_kind


def output_schema(event_id_type: pa.DataType, id_type: pa.DataType) -> pa.Schema:
    return pa.schema(
        [
            ("id", id_type),
            ("adm1", pa.string()),
            ("event_id", event_id_type),
            ("rp", pa.float32()),
            ("max_depth_m", pa.float32()),
            ("max_depth_defended_m", pa.float32()),
            ("exposed_size", pa.float64()),
            ("exposed_size_defended", pa.float64()),
            ("damage_undefended_usd", pa.float64()),
            ("damage_defended_usd", pa.float64()),
        ]
    )


@click.command()
@click.option("--haz_id", required=True)
@click.option("--layer", "layers", required=True, multiple=True)
@click.option("--type", "event_types", required=True, multiple=True)
@click.option("--splits_template", required=True, help="Piece table path, with {layer}")
@click.option("--haz_event_rp_template", required=True, help="(HAZ, event) RP path, with {TYPE}")
@click.option("--rp", "rps", required=True, multiple=True, type=float)
@click.option("--horp", required=True, type=float, help="Hazard onset return period")
@click.option("--cost_template", required=True, help="Cost table path, with {layer}")
@click.option("--curves_template", required=True, help="Damage curve directory, with {asset_type}")
@click.option("--output_template", required=True, help="Output path, with {TYPE} and {layer}")
def main(haz_id, layers, event_types, splits_template, haz_event_rp_template, rps, horp,
         cost_template, curves_template, output_template):
    events = {}
    for event_type in event_types:
        path = haz_event_rp_template.format(TYPE=event_type)
        table = pd.read_parquet(path, filters=[(HAZ_ID, "==", str(haz_id))], columns=[EVENT_ID, "rp"])
        events[event_type] = (table, pq.read_schema(path).field(EVENT_ID).type)
        logging.info("%s: %d events for HAZ %s", event_type, len(table), haz_id)

    for layer in layers:
        splits_path = splits_template.format(layer=layer)
        depth_cols = [f"depth_rp{rp:g}" for rp in sorted(rps)]
        splits = pd.read_parquet(
            splits_path, columns=["split_id", "id", "adm1", "cost_type", "size", "sop", *depth_cols]
        )
        costing = PieceCosting(
            splits, rps, horp, layer_kind(layer), layer,
            cost_table=cost_template.format(layer=layer),
            curves_dir=curves_template.format(asset_type=layer_asset_type(layer)),
        )
        id_type = pq.read_schema(splits_path).field("id").type
        for event_type, (table, event_id_type) in events.items():
            output = output_template.format(TYPE=event_type, layer=layer)
            os.makedirs(os.path.dirname(output), exist_ok=True)
            schema = output_schema(event_id_type, id_type)
            n_rows = 0
            with pq.ParquetWriter(output, schema) as writer:
                for chunk in asset_event_chunks(costing, table, EVENT_ID):
                    writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
                    n_rows += len(chunk)
                if n_rows == 0:
                    writer.write_table(schema.empty_table())
            logging.info("%s %s: %d wet pieces, wrote %d (asset, event) rows", event_type, layer, len(costing), n_rows)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
