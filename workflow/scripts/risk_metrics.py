"""Expected annual values and exceedance values, at asset, HAZ, adm1 and adm0 level.

For one layer, give `--asset_events` and one `--event_losses_dir`. For totals
over several layers, give several `--event_losses_dir` and no `--asset_events`
(asset-level metrics are then skipped). See `jba_events.risk` for the method.

For each level this writes:

    {level}_expected_annual.parquet   key..., expected_annual_{value}...
    {level}_exceedance.parquet        key..., probability, return_period, {value}...

Example:

    python workflow/scripts/risk_metrics.py --years 10000 \
        --probability 0.01 --probability 0.001 \
        --value damage_undefended_usd --value damage_defended_usd \
        --asset_events processed_data/asset_event_damages/Sim/road_edges \
        --event_losses_dir processed_data/event_losses/Sim/road_edges \
        --output_dir processed_data/risk/road_edges
"""

import logging
import os

import click
import duckdb

from jba_events.risk import exceedance, expected_annual

LEVEL_KEYS = {"haz": ["haz"], "adm1": ["adm1", "adm0"], "adm0": ["adm0"]}


@click.command()
@click.option("--years", required=True, type=float, help="Years represented by the event set")
@click.option("--probability", "probabilities", required=True, multiple=True, type=float)
@click.option("--value", "values", required=True, multiple=True)
@click.option("--asset_events", default=None, type=click.Path(exists=True, file_okay=False))
@click.option("--event_losses_dir", "loss_dirs", required=True, multiple=True, type=click.Path(exists=True, file_okay=False))
@click.option("--output_dir", required=True, type=click.Path(file_okay=False))
def main(years, probabilities, values, asset_events, loss_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    con = duckdb.connect()
    values = list(values)
    sums = ", ".join(f'sum("{v}") AS "{v}"' for v in values)

    relations = {}
    if asset_events:
        # an asset can have rows in several HAZs and admin regions for one event
        pattern = os.path.join(asset_events, "haz=*", "*.parquet").replace("'", "''")
        con.sql(
            f"CREATE TEMP TABLE asset_events AS SELECT id, event_id, {sums} "
            f"FROM read_parquet('{pattern}') GROUP BY id, event_id"
        )
        relations["asset"] = (["id"], "asset_events")
    for level, keys in LEVEL_KEYS.items():
        files = ", ".join("'" + os.path.join(d, f"{level}.parquet").replace("'", "''") + "'" for d in loss_dirs)
        key_sql = ", ".join(keys)
        con.sql(
            f"CREATE TEMP TABLE {level}_events AS SELECT {key_sql}, event_id, {sums} "
            f"FROM read_parquet([{files}]) GROUP BY {key_sql}, event_id"
        )
        relations[level] = (keys, f"{level}_events")

    for level, (keys, relation) in relations.items():
        ead = expected_annual(con, relation, keys, values, years)
        ead.to_parquet(os.path.join(output_dir, f"{level}_expected_annual.parquet"), index=False)
        exc = exceedance(con, relation, keys, values, years, probabilities)
        exc.to_parquet(os.path.join(output_dir, f"{level}_exceedance.parquet"), index=False)
        logging.info("%s: %d keys", level, len(ead))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
