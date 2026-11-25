"""Static criticality of road assets, for one chunk of the assets.

For each asset, independent of any flood: the population whose baseline route
to its nearest urban centre uses the asset, and what happens to them if the
asset alone fails - how many are disconnected from every centre, how many
are detoured, and the extra person-km.

Output rows (only assets some population routes through): asset_id,
population_routed, population_disconnected, population_detoured, detour_person_km.

Example:

    python workflow/scripts/network_criticality.py \
        --baseline_dir processed_data/disconnection/baseline --chunk 0 --n_chunks 20 \
        --output processed_data/disconnection/criticality/part-0.parquet
"""

import logging
import os
import click
import numpy as np
import pandas as pd

from jba_events.disconnection import RouteNetwork

COLUMNS = ["asset_id", "population_routed", "population_disconnected", "population_detoured", "detour_person_km"]


@click.command()
@click.option("--baseline_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--chunk", required=True, type=int)
@click.option("--n_chunks", required=True, type=int)
@click.option("--output", required=True, type=click.Path(dir_okay=False))
def main(baseline_dir, chunk, n_chunks, output):
    network = RouteNetwork.from_baseline(baseline_dir)
    asset_ids = network.asset_ids.to_numpy(dtype=str)
    mine = pd.util.hash_array(asset_ids.astype(object)) % np.uint64(n_chunks) == np.uint64(chunk)
    logging.info("Chunk %d of %d: %d of %d assets", chunk, n_chunks, mine.sum(), len(asset_ids))

    rows = []
    for i in np.flatnonzero(mine):
        failed = network.asset_edge_slice(i)
        intervals = network.affected_intervals(failed)
        if len(intervals) == 0:
            continue
        routed = network.population_at_risk(intervals)
        if routed == 0:
            continue
        total = network.impact(failed, by_admin=False)
        values = total.iloc[0].to_dict() if len(total) else {}
        rows.append(
            (
                asset_ids[i],
                routed,
                values.get("population_disconnected", 0.0),
                values.get("population_detoured", 0.0),
                values.get("detour_person_km", 0.0),
            )
        )
    out = pd.DataFrame(rows, columns=COLUMNS).astype({c: float for c in COLUMNS[1:]})
    os.makedirs(os.path.dirname(output), exist_ok=True)
    out.to_parquet(output, index=False)
    logging.info("Wrote %d assets to %s", len(out), output)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
