"""Link every HAZ to its river observation points (OPs).

Example:

    python workflow/scripts/link_haz_op.py \
        --haz_path processed_data/basins/haz_500.gpkg --haz_id_column T500_ID \
        --op_path processed_data/events/RiverOpInfo.gpkg \
        --max_distance_m 100000 --distance_crs ESRI:102023 \
        --output processed_data/prepared/events/haz_op_links.parquet
"""

import logging

import click
import geopandas as gpd

from floodlib.events import OP_ID, link_haz_op


@click.command()
@click.option("--haz_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--haz_id_column", required=True)
@click.option("--op_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--max_distance_m", required=True, type=float)
@click.option("--distance_crs", required=True)
@click.option("--output", required=True, type=click.Path(dir_okay=False, writable=True))
def main(haz_path, haz_id_column, op_path, max_distance_m, distance_crs, output):
    haz = gpd.read_file(haz_path, columns=[haz_id_column])
    haz[haz_id_column] = haz[haz_id_column].astype(str)
    ops = gpd.read_file(op_path, columns=[OP_ID])
    links = link_haz_op(haz, ops, haz_id_column, max_distance_m, distance_crs)
    links.to_parquet(output, index=False)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
