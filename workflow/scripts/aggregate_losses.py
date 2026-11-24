"""Event losses for one layer and event set, summed to HAZ, admin-1 and admin-0.

Reads the per-HAZ (asset, adm1, event) tables written by `event_damages.py`
and writes `haz.parquet`, `adm1.parquet` and `adm0.parquet`, each with one row
per (region, event). An event that spans several HAZs has one row per HAZ in
`haz.parquet`, but a single summed row per region in the admin tables.

Example:

    python workflow/scripts/aggregate_losses.py \
        --asset_events processed_data/asset_event_damages/Sim/road_edges \
        --admin_path processed_data/admin/adm1.gpkg --adm1_column GID_1 --adm0_column GID_0 \
        --output_dir processed_data/event_losses/Sim/road_edges
"""

import logging
import os

import click
import duckdb
import pandas as pd
import pyogrio

VALUES = ["exposed_size", "exposed_size_defended", "damage_undefended_usd", "damage_defended_usd"]


def asset_events_relation(directory: str) -> str:
    """DuckDB table expression for a directory of `haz=*/part.parquet` files."""
    pattern = os.path.join(directory, "haz=*", "*.parquet").replace("'", "''")
    return f"read_parquet('{pattern}', hive_partitioning = true, hive_types_autocast = false)"


def read_adm_lookup(admin_path, adm1_column, adm0_column) -> pd.DataFrame:
    if not admin_path:
        return pd.DataFrame({"adm1": pd.Series(dtype=str), "adm0": pd.Series(dtype=str)})
    table = pyogrio.read_dataframe(admin_path, columns=[adm1_column, adm0_column], read_geometry=False)
    table = table.rename(columns={adm1_column: "adm1", adm0_column: "adm0"}).astype(str)
    return table.drop_duplicates("adm1")


@click.command()
@click.option("--asset_events", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--admin_path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--adm1_column", default=None)
@click.option("--adm0_column", default=None)
@click.option("--output_dir", required=True, type=click.Path(file_okay=False))
def main(asset_events, admin_path, adm1_column, adm0_column, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    con = duckdb.connect()
    con.register("adm_lookup", read_adm_lookup(admin_path, adm1_column, adm0_column))
    source = asset_events_relation(asset_events)
    sums = ", ".join(f"sum({v}) AS {v}" for v in VALUES)

    queries = {
        "haz": f"SELECT haz, event_id, {sums} FROM {source} GROUP BY haz, event_id ORDER BY haz, event_id",
        "adm1": f"""
            SELECT s.adm1, l.adm0, event_id, {sums}
            FROM {source} s LEFT JOIN adm_lookup l ON s.adm1 = l.adm1
            GROUP BY s.adm1, l.adm0, event_id ORDER BY s.adm1, event_id""",
        "adm0": f"""
            SELECT l.adm0, event_id, {sums}
            FROM {source} s LEFT JOIN adm_lookup l ON s.adm1 = l.adm1
            GROUP BY l.adm0, event_id ORDER BY l.adm0, event_id""",
    }
    for level, query in queries.items():
        output = os.path.join(output_dir, f"{level}.parquet").replace("'", "''")
        con.sql(f"COPY ({query}) TO '{output}' (FORMAT parquet)")
        logging.info("Wrote %s", output)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
