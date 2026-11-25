"""Combine criticality chunks, ranked, with each asset's chance of failing in a flood.

Writes one row per asset that some population routes through: the columns of
`network_criticality.py`, plus annual_failure_frequency_undefended and
_defended (events per year in which the asset fails, from the Sim set), and
the expected annual population disconnected and person-km of detour if the
asset failed alone in those events (criticality x frequency). Sorted by
population disconnected, then detour person-km.

Example:

    python workflow/scripts/network_criticality_summary.py \
        --asset_events processed_data/asset_event_damages/Sim/road_edges \
        --threshold_m 0.5 --years 10000 \
        --output processed_data/disconnection/criticality.parquet \
        processed_data/disconnection/criticality/part-*.parquet
"""

import logging
import os

import click
import duckdb


@click.command()
@click.option("--asset_events", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--threshold_m", required=True, type=float)
@click.option("--years", required=True, type=float)
@click.option("--output", required=True, type=click.Path(dir_okay=False))
@click.argument("parts", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
def main(asset_events, threshold_m, years, output, parts):
    files = ", ".join("'" + p.replace("'", "''") + "'" for p in parts)
    pattern = os.path.join(asset_events, "haz=*", "*.parquet").replace("'", "''")
    out = output.replace("'", "''")
    duckdb.sql(
        f"""
        COPY (
            WITH failures AS (
                SELECT CAST(id AS VARCHAR) AS asset_id,
                    count(DISTINCT CASE WHEN max_depth_m >= {threshold_m} THEN event_id END) / {years}
                        AS annual_failure_frequency_undefended,
                    count(DISTINCT CASE WHEN max_depth_defended_m >= {threshold_m} THEN event_id END) / {years}
                        AS annual_failure_frequency_defended
                FROM read_parquet('{pattern}')
                GROUP BY id
            )
            SELECT c.*,
                coalesce(f.annual_failure_frequency_undefended, 0) AS annual_failure_frequency_undefended,
                coalesce(f.annual_failure_frequency_defended, 0) AS annual_failure_frequency_defended,
                c.population_disconnected * coalesce(f.annual_failure_frequency_undefended, 0)
                    AS expected_annual_population_disconnected_if_alone,
                c.detour_person_km * coalesce(f.annual_failure_frequency_undefended, 0)
                    AS expected_annual_detour_person_km_if_alone
            FROM read_parquet([{files}]) c LEFT JOIN failures f USING (asset_id)
            ORDER BY population_disconnected DESC, detour_person_km DESC, asset_id
        ) TO '{out}' (FORMAT parquet)
        """
    )
    logging.info("Wrote %s", output)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
