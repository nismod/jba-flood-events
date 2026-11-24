"""Return period for each (HAZ, event), from event RPs at observation points.

The output is sorted by HAZ, so per-HAZ jobs can read just their rows with a
parquet filter.

Example:

    python workflow/scripts/haz_event_rp.py \
        --events_path processed_data/events/SimEventRp.csv \
        --links_path processed_data/prepared/events/haz_op_links.parquet \
        --output processed_data/prepared/events/Sim_haz_event_rp.parquet
"""

import logging

import click
import pandas as pd

from floodlib.events import EVENT_ID, HAZ_ID, OP_ID, haz_event_rp


@click.command()
@click.option("--events_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--links_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", required=True, type=click.Path(dir_okay=False, writable=True))
def main(events_path, links_path, output):
    events = pd.read_csv(events_path, usecols=[EVENT_ID, OP_ID, "rp"])
    dupes = events.duplicated([EVENT_ID, OP_ID])
    if dupes.any():
        raise ValueError(
            f"{events_path}: {dupes.sum()} duplicate ({EVENT_ID}, {OP_ID}) rows. "
            "If several regional event files were concatenated, event IDs may not be unique."
        )
    links = pd.read_parquet(links_path)
    rp = haz_event_rp(events, links)
    logging.info(
        "%d (HAZ, event) pairs over %d HAZs and %d events",
        len(rp),
        rp[HAZ_ID].nunique(),
        rp[EVENT_ID].nunique(),
    )
    rp.to_parquet(output, index=False, row_group_size=100_000)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
