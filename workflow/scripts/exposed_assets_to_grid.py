import click
import os
import logging
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from pyproj import Geod
import pathlib
import numpy as np
import tempfile
import rioxarray as riox
import rasterio
import snail.intersection as snint
from shapely.geometry import MultiLineString, MultiPolygon
import shapely



@click.command()
@click.version_option("1.0")

@click.option(
    "--rp_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    help="Path to RP TIFF",
)
@click.option(
    "--asset_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    help="Path to asset geoparquet",
)
@click.option(
    "--output_path",
    required=True,
    type=click.Path(exists=False, dir_okay=False, file_okay=True, writable=True),
    help="Path to exposed asset output file",
)



def main(rp_path, asset_path, output_path):
    """
    Example usage:

        python workflow/scripts/exposed_assets_to_grid.py \
            --rp_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/flrf_ud_Q1500.tif \
            --asset_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/maritime_polygons_network.geoparquet\
            --output_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/exposed_maritime_polygons_network.geoparquet
    """

    logging.info("Reading raster and asset data...")
    

    # upload input data

    crs = "EPSG:4326"
    vector = gpd.read_parquet(asset_path)
    vector = vector.to_crs(crs)
    if len(vector) == 0:
        logging.info("No assets in this HAZ, writing empty file.")
        vector.to_parquet(output_path)
        return
    vector['geometry'] = vector['geometry'].apply(
            lambda g: shapely.force_2d(g)  # ensure 2D
        )
    vector = vector.explode(index_parts=False)  # MultiLineString -> LineString etc.
    vector = vector.reset_index(drop=True)
    print("vector types:", vector.geometry.geom_type.value_counts().to_dict())

    # Check for invalid/null geometries
    print("Total rows:", len(vector))
    print("Null geometries:", vector.geometry.isna().sum())
    print("Invalid geometries:", (~vector.geometry.is_valid).sum())
    print(vector.total_bounds)
    
    with rasterio.open(rp_path) as src:
        bounds = src.bounds
    
    grid, window = grid_from_window(rp_path, bounds)


    # first feature geometry
    geom = vector.geometry.iat[0]

    
    
    # type if one feature
    if geom.geom_type == "Point":
        
        logging.info("Processing point geometries...")
        
        vector = vector.reset_index(drop=True)
        vector_splits = vector.copy()  # No splitting needed for points

    elif geom.geom_type == "LineString":
            
        logging.info("Splitting edges...")
        vector = snint.prepare_linestrings(vector)
        vector = vector.reset_index(drop=True)
        vector_splits = snint.split_linestrings(vector, grid)
        logging.info("Split %d edges into %d pieces", len(vector), len(vector_splits))
 
    elif geom.geom_type == "Polygon":
        
        logging.info("Splitting polygons...")
        vector = vector.reset_index(drop=True)
        vector_splits = snint.split_polygons(vector, grid)
        logging.info("Split %d polygons into %d pieces", len(vector), len(vector_splits))
    
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")
    
    logging.info("Finding indices...")
    vector_splits = snint.apply_indices(
        vector_splits, grid, index_i="raster_i", index_j="raster_j"
    )
    vector_splits.to_parquet(output_path)

    logging.info("Done.")




def grid_from_window(raster_file, bounds, verbose=False):
    with rasterio.open(raster_file) as src:
        window = rasterio.windows.from_bounds(
            bounds[0], bounds[1], bounds[2], bounds[3],
            transform=src.transform
        ).round()
        logging.info(f"Computed window from bounds: {window}")
        window_transform = rasterio.windows.transform(window, src.transform)
    transform_6 = tuple(window_transform)[:6]

    grid = snint.GridDefinition(
        width=int(window.width),
        height=int(window.height),
        transform=transform_6,
        crs=src.crs.to_string()
    )
    return grid, window

def process_raster_grid(
        raster_files:list[str], vector:gpd.GeoDataFrame, verify_consistency=False
        ) -> snint.GridDefinition:
    """Make a grid for list of rasters, based on vector bounds."""
    bounds = vector.total_bounds
    grid, window = grid_from_window(raster_files[0], bounds)
    logging.info(f"{grid=}")

    if len(raster_files) > 1 and verify_consistency:
        logging.info("Checking raster grid consistency")
        for raster_path in raster_files[1:]:
            other_grid, _ = grid_from_window(raster_path, bounds)
            if other_grid != grid:
                raise AttributeError(
                    (
                        f"Raster attribute mismatch in file {raster_path}:\n"
                        f"Height: expected={grid.height}; actual={other_grid.height}\n"
                        f"Width: expected={grid.width}; actual={other_grid.width}\n"
                        f"Transform equal? {other_grid.transform == grid.transform}\n"
                        f"Transform expected= {grid.transform}\n"
                        f"Transform actual= {other_grid.transform}\n"
                        f"CRS equal? {other_grid.crs == grid.crs}"
                    )
                )
    
    return grid, window

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s",
        level=logging.INFO
    )

    main()