"""Check the workflow inputs before any heavy job runs.

Collects every problem found and fails with the full list, so data issues
show up in the first minute rather than part-way through a cluster run.
Writes a short JSON report on success.

Example:

    python workflow/scripts/validate_inputs.py \
        --haz_path processed_data/basins/haz_500.gpkg --haz_id_column T500_ID \
        --op_path processed_data/events/RiverOpInfo.gpkg \
        --events_path processed_data/events/ObsEventRp.csv --events_path processed_data/events/SimEventRp.csv \
        --layer road_edges --asset_template 'processed_data/infrastructure/africa_{layer}_network.parquet' \
        --id_column id --cost_template 'processed_data/costs/{layer}.csv' \
        --curves_template 'processed_data/curves/{asset_type}' \
        --output processed_data/prepared/validation.json
"""

import json
import logging
import os

import click
import pandas as pd
import pyarrow.parquet as pq
import pyogrio

from jba_events.damage import COST_COLUMNS
from jba_events.events import EVENT_ID, OP_ID
from jba_events.split import ALLOWED_TYPES, layer_asset_type, layer_kind


def vector_columns(path) -> set:
    return set(pyogrio.read_info(path)["fields"])


def check_vector(errors, path, columns, name):
    missing = [c for c in columns if c and c not in vector_columns(path)]
    if missing:
        errors.append(f"{name} ({path}): missing columns {missing}")
    return not missing


def check_layer(errors, layer, path, id_column, cost_table, curves_dir) -> dict:
    kind = layer_kind(layer)
    if not os.path.exists(path):
        errors.append(f"{layer}: no source file {path}")
        return {}
    schema = pq.read_schema(path)
    if id_column not in schema.names:
        errors.append(f"{layer} ({path}): no '{id_column}' column")
        return {}
    ids = pq.read_table(path, columns=[id_column]).column(id_column).to_pandas()
    if ids.isna().any() or ids.duplicated().any():
        errors.append(f"{layer}: '{id_column}' has {ids.isna().sum()} nulls and {ids.duplicated().sum()} duplicates")

    geo = json.loads(schema.metadata.get(b"geo", b"{}")) if schema.metadata else {}
    for column in geo.get("columns", {}).values():
        types = {t.replace(" Z", "") for t in column.get("geometry_types", [])}
        if types - ALLOWED_TYPES[kind]:
            errors.append(f"{layer}: geometry types {sorted(types)} don't match a {kind} layer")

    report = {"features": len(ids), "cost_types": []}
    if "cost_type" not in schema.names:
        return report
    cost_types = sorted(pq.read_table(path, columns=["cost_type"]).column("cost_type").to_pandas().dropna().unique())
    report["cost_types"] = cost_types
    if not cost_types:
        return report
    if not os.path.exists(cost_table):
        errors.append(f"{layer}: has cost_types {cost_types} but no cost table {cost_table}")
    else:
        costs = pd.read_csv(cost_table)
        needed = {"cost_type", COST_COLUMNS[kind]}
        if needed - set(costs.columns):
            errors.append(f"{cost_table}: missing columns {sorted(needed - set(costs.columns))}")
        elif set(cost_types) - set(costs.cost_type):
            errors.append(f"{cost_table}: no unit cost for {sorted(set(cost_types) - set(costs.cost_type))}")
    for cost_type in cost_types:
        curve = os.path.join(curves_dir, f"{cost_type}.csv")
        if not os.path.exists(curve):
            errors.append(f"{layer}: no damage curve {curve}")
        elif {"flood_depth_m", "damage_fraction_mean"} - set(pd.read_csv(curve, comment="#").columns):
            errors.append(f"{curve}: needs columns flood_depth_m and damage_fraction_mean")
    return report


@click.command()
@click.option("--haz_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--haz_id_column", required=True)
@click.option("--op_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--events_path", "events_paths", multiple=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--layer", "layers", multiple=True)
@click.option("--asset_template", required=True)
@click.option("--id_column", default="id", show_default=True)
@click.option("--cost_template", required=True)
@click.option("--curves_template", required=True)
@click.option("--defended_path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--sop_column", default=None)
@click.option("--admin_path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--adm1_column", default=None)
@click.option("--adm0_column", default=None)
@click.option("--output", required=True, type=click.Path(dir_okay=False))
def main(haz_path, haz_id_column, op_path, events_paths, layers, asset_template, id_column, cost_template,
         curves_template, defended_path, sop_column, admin_path, adm1_column, adm0_column, output):
    errors = []
    report = {"layers": {}, "events": {}}

    if check_vector(errors, haz_path, [haz_id_column], "HAZ"):
        haz_ids = pyogrio.read_dataframe(haz_path, columns=[haz_id_column], read_geometry=False)[haz_id_column]
        report["hazs"] = len(haz_ids)
        if haz_ids.duplicated().any():
            errors.append(f"HAZ: {haz_ids.duplicated().sum()} duplicate {haz_id_column} values")
    check_vector(errors, op_path, [OP_ID], "OPs")
    if defended_path:
        check_vector(errors, defended_path, [sop_column], "Defended areas")
    if admin_path:
        check_vector(errors, admin_path, [adm1_column, adm0_column], "Admin")

    for path in events_paths:
        header = pd.read_csv(path, nrows=0).columns
        if {EVENT_ID, OP_ID, "rp"} - set(header):
            errors.append(f"{path}: needs columns {EVENT_ID}, {OP_ID}, rp")
            continue
        events = pd.read_csv(path, usecols=[EVENT_ID, OP_ID, "rp"])
        dupes = int(events.duplicated([EVENT_ID, OP_ID]).sum())
        if dupes:
            errors.append(
                f"{path}: {dupes} duplicate ({EVENT_ID}, {OP_ID}) rows; "
                "event IDs may not be unique across concatenated regional files"
            )
        if (events.rp <= 0).any() or events.rp.isna().any():
            errors.append(f"{path}: rp has nulls or non-positive values")
        report["events"][path] = {"events": int(events[EVENT_ID].nunique()), "rows": len(events)}

    if not layers:
        errors.append("No asset layers found; check assets.source_path and assets.layers")
    for layer in layers:
        report["layers"][layer] = check_layer(
            errors, layer, asset_template.format(layer=layer), id_column,
            cost_template.format(layer=layer), curves_template.format(asset_type=layer_asset_type(layer)),
        )

    if errors:
        raise click.ClickException("Input validation failed:\n- " + "\n- ".join(errors))
    with open(output, "w") as fh:
        json.dump(report, fh, indent=2)
    logging.info("Inputs OK: %s", report)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
