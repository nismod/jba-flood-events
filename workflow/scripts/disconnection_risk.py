"""Annual probability and expectation of population disconnected, by region.

Reads the per-event tables from `disconnection_events.py` and writes, for each
of the levels adm1, adm0 and total:

    {level}_expected_annual.parquet
        key..., protection, annual_probability_any (of at least one event that
        disconnects anyone), expected_annual_population_disconnected,
        expected_annual_population_isolated (people x events per year)
    {level}_exceedance.parquet
        key..., protection, probability, return_period,
        population_disconnected, population_isolated

See `jba_events.risk` for the method.

Example:

    python workflow/scripts/disconnection_risk.py --years 10000 --probability 0.01 \
        --output_dir processed_data/disconnection/Sim/risk \
        processed_data/disconnection/Sim/events/part-*.parquet
"""

import logging
import math
import os

import click
import duckdb

from jba_events.risk import exceedance, expected_annual

VALUES = ["population_disconnected", "population_isolated"]
LEVEL_KEYS = {"adm1": ["adm1", "adm0"], "adm0": ["adm0"], "total": []}


@click.command()
@click.option("--years", required=True, type=float)
@click.option("--probability", "probabilities", required=True, multiple=True, type=float)
@click.option("--output_dir", required=True, type=click.Path(file_okay=False))
@click.argument("event_tables", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
def main(years, probabilities, output_dir, event_tables):
    os.makedirs(output_dir, exist_ok=True)
    con = duckdb.connect()
    files = ", ".join("'" + f.replace("'", "''") + "'" for f in event_tables)
    sums = ", ".join(f"sum({v}) AS {v}" for v in VALUES)

    for level, region_keys in LEVEL_KEYS.items():
        keys = [*region_keys, "protection"]
        key_sql = ", ".join(keys)
        con.sql(
            f"CREATE OR REPLACE TEMP TABLE events AS SELECT {key_sql}, event_id, {sums} "
            f"FROM read_parquet([{files}], union_by_name = true) GROUP BY {key_sql}, event_id"
        )
        expected = expected_annual(con, "events", keys, VALUES, years)
        counts = con.sql(
            f"SELECT {key_sql}, count(*) AS n_events FROM events WHERE population_disconnected > 0 GROUP BY {key_sql}"
        ).df()
        expected = expected.merge(counts, on=keys, how="left")
        expected["annual_probability_any"] = [
            1 - math.exp(-n / years) for n in expected.pop("n_events").fillna(0)
        ]
        expected.to_parquet(os.path.join(output_dir, f"{level}_expected_annual.parquet"), index=False)
        exceedance(con, "events", keys, VALUES, years, probabilities).to_parquet(
            os.path.join(output_dir, f"{level}_exceedance.parquet"), index=False
        )
        logging.info("%s: %d rows", level, len(expected))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
