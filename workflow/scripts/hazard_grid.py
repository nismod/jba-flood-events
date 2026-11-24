"""Check all return period maps share one grid, and record that grid.

Example:

    python workflow/scripts/hazard_grid.py \
        --output processed_data/prepared/hazard_grid.json \
        processed_data/hazards/flrf_ud_Q20.tif processed_data/hazards/flrf_ud_Q50.tif ...
"""

import logging

import click

from floodlib.grid import HazardGrid


@click.command()
@click.option("--output", required=True, type=click.Path(dir_okay=False, writable=True))
@click.argument("rp_paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
def main(output, rp_paths):
    grid = HazardGrid.from_rasters(rp_paths)
    logging.info("Hazard grid: %s", grid)
    grid.to_json(output)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
