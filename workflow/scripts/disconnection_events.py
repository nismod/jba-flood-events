"""Population disconnected and detoured per event, for one chunk of an event set.

An asset fails in an event if its maximum flood depth reaches `--threshold_m`.
The undefended case uses the undefended depth; the defended case ignores
pieces protected by a standard of protection at or above the event RP.

Output rows: event_id, protection (undefended|defended), adm1, adm0,
population_disconnected, population_detoured, detour_person_km. Only regions
with some impact have rows; adm1 and adm0 are those of the population cell.

Example:

    python workflow/scripts/disconnection_events.py \
        --baseline_dir processed_data/disconnection/baseline \
        --asset_events processed_data/asset_event_damages/Sim/road_edges \
        --threshold_m 0.5 --chunk 0 --n_chunks 20 \
        --output processed_data/disconnection/Sim/events/part-0.parquet
"""

import logging
import os

import click
import duckdb
import pandas as pd

from jba_events.disconnection import RouteNetwork

VALUES = ["population_disconnected", "population_detoured", "detour_person_km"]
COLUMNS = ["event_id", "protection", "adm1", "adm0", *VALUES]


@click.command()
@click.option("--baseline_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--asset_events", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--threshold_m", required=True, type=float)
@click.option("--chunk", required=True, type=int)
@click.option("--n_chunks", required=True, type=int)
@click.option("--output", required=True, type=click.Path(dir_okay=False))
def main(baseline_dir, asset_events, threshold_m, chunk, n_chunks, output):
    network = RouteNetwork.from_baseline(baseline_dir)
    pattern = os.path.join(asset_events, "haz=*", "*.parquet").replace("'", "''")
    failed = duckdb.sql(
        f"""
        SELECT event_id, id, max(max_depth_defended_m) >= {threshold_m} AS fails_defended
        FROM read_parquet('{pattern}')
        WHERE hash(event_id) % {n_chunks} = {chunk} AND max_depth_m >= {threshold_m}
        GROUP BY event_id, id
        ORDER BY event_id
        """
    ).df()
    logging.info("Chunk %d of %d: %d events with failed assets", chunk, n_chunks, failed.event_id.nunique())

    frames = []
    for event_id, group in failed.groupby("event_id", sort=False):
        ids = group["id"].to_numpy()
        defended_ids = ids[group.fails_defended.to_numpy()]
        undefended = network.impact(network.edges_of_assets(ids))
        if len(defended_ids) == len(ids):
            defended = undefended
        elif len(defended_ids):
            defended = network.impact(network.edges_of_assets(defended_ids))
        else:
            defended = undefended.iloc[0:0]
        for protection, table in [("undefended", undefended), ("defended", defended)]:
            if len(table):
                frames.append(table.assign(event_id=event_id, protection=protection))

    if frames:
        out = pd.concat(frames, ignore_index=True)[COLUMNS]
    else:
        out = pd.DataFrame({c: pd.Series(dtype=float if c in VALUES else object) for c in COLUMNS})
        out["event_id"] = out["event_id"].astype(failed.event_id.dtype)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    out.to_parquet(output, index=False)
    logging.info("Wrote %d rows to %s", len(out), output)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
