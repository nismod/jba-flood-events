"""Summarise exposure per layer and HAZ, as a check on the split step.

One row per (layer, HAZ): assets, pieces, total size, size in defended areas,
and size wet at each mapped RP. Compare the totals against the source layers
and look for HAZs with unexpectedly little or much exposure.

Example:

    python workflow/scripts/qa_summary.py --rp 20 --rp 50 \
        --output processed_data/qa/exposure_summary.csv \
        road_edges=processed_data/exposure/road_edges railway_edges=processed_data/exposure/railway_edges
"""

import logging
import os

import click
import duckdb
import pandas as pd


@click.command()
@click.option("--rp", "rps", required=True, multiple=True, type=float)
@click.option("--output", required=True, type=click.Path(dir_okay=False))
@click.argument("layer_dirs", nargs=-1, required=True)
def main(rps, output, layer_dirs):
    wet = ", ".join(
        f"sum(CASE WHEN \"depth_rp{rp:g}\" > 0 THEN size ELSE 0 END) AS \"size_wet_rp{rp:g}\"" for rp in rps
    )
    frames = []
    for item in layer_dirs:
        layer, directory = item.split("=", 1)
        pattern = os.path.join(directory, "haz=*", "*.parquet").replace("'", "''")
        frames.append(
            duckdb.sql(
                f"""
                SELECT '{layer}' AS layer, haz, any_value(size_unit) AS size_unit,
                    count(DISTINCT id) AS assets, count(*) AS pieces, sum(size) AS size,
                    sum(CASE WHEN sop IS NOT NULL AND NOT isnan(sop) THEN size ELSE 0 END) AS size_defended,
                    {wet}
                FROM read_parquet('{pattern}', hive_partitioning = true, hive_types_autocast = false,
                                  union_by_name = true)
                GROUP BY haz ORDER BY haz
                """
            ).df()
        )
    summary = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    summary.to_csv(output, index=False)
    logging.info("Wrote %d rows to %s", len(summary), output)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
