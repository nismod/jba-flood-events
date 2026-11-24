"""Prepare a continental asset layer for fast bounding-box reads.

The output is in the hazard grid CRS, single-part, sorted along a Hilbert
curve and written as GeoParquet with a bbox covering column. A read with
`gpd.read_parquet(path, bbox=...)` then only touches nearby row groups.

Example:

    python workflow/scripts/prepare_assets.py \
        --layer road_edges \
        --asset_path processed_data/infrastructure/africa_road_edges_network.parquet \
        --grid processed_data/prepared/hazard_grid.json \
        --output processed_data/prepared/assets/road_edges.parquet
"""

import logging

import click
import geopandas as gpd
import numpy as np
import shapely

from jba_events.grid import HazardGrid
from jba_events.split import check_geometry_types, layer_kind

ROW_GROUP_SIZE = 20_000


@click.command()
@click.option("--layer", required=True)
@click.option("--asset_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--grid", "grid_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--id_column", default="id", show_default=True)
@click.option("--output", required=True, type=click.Path(dir_okay=False, writable=True))
def main(layer, asset_path, grid_path, id_column, output):
    kind = layer_kind(layer)
    grid = HazardGrid.from_json(grid_path)

    assets = gpd.read_parquet(asset_path)
    logging.info("Read %d features from %s", len(assets), asset_path)
    if id_column not in assets.columns:
        raise ValueError(f"{asset_path}: no '{id_column}' column")
    if assets[id_column].isna().any() or assets[id_column].duplicated().any():
        raise ValueError(f"{asset_path}: '{id_column}' must be unique and not null")
    if id_column != "id":
        assets = assets.rename(columns={id_column: "id"})
    if "cost_type" not in assets.columns:
        logging.warning("%s: no cost_type column, assets will be exposure-only", layer)
        assets["cost_type"] = None

    assets = assets[~(assets.geometry.isna() | assets.geometry.is_empty)]
    assets = assets.to_crs(grid.crs)
    assets.geometry = shapely.force_2d(assets.geometry.values)
    if kind == "polygon":
        invalid = ~assets.geometry.is_valid
        if invalid.any():
            logging.info("Repairing %d invalid polygons", invalid.sum())
            assets.loc[invalid, "geometry"] = assets.geometry[invalid].make_valid()
        # make_valid can return collections; keep only polygonal parts
        assets = assets.explode(ignore_index=True)
        assets = assets[assets.geometry.geom_type == "Polygon"]
    else:
        assets = assets.explode(ignore_index=True)
    check_geometry_types(assets, kind, layer)

    if len(assets):
        order = np.argsort(assets.geometry.hilbert_distance(total_bounds=assets.total_bounds).values)
        assets = assets.iloc[order].reset_index(drop=True)

    assets.to_parquet(output, write_covering_bbox=True, row_group_size=ROW_GROUP_SIZE)
    logging.info("Wrote %d single-part features to %s", len(assets), output)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
