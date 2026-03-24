
import logging

import click
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as riox
import os
from shapely.geometry import Polygon

@click.command()
@click.version_option("1.0")
@click.option(
    "--haz_id",
    required=True,
    type=str,
    help="ID of the HAZ to clip to, e.g. 001"
)
@click.option(
    "--haz_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    help="Path to HAZ GeoPackage",
)
@click.option(
    "--asset_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    help="Path to asset GeoParquet",
)
@click.option(
    "--output_path",
    required=True,
    type=click.Path(exists=False, dir_okay=False, file_okay=True, writable=True),
    help="Path to clipped output file",
)

def main(haz_id, haz_path, asset_path, output_path):
    """
    Example usage:

        python workflow/scripts/clip_asset_to_HAZ.py \
            --haz_id 500_13_19495 \
            --haz_path ~/Desktop/DataFolders/JBA_flooding/processed_data/basins/haz_500.gpkg \
            --asset_path ~/Desktop/DataFolders/JBA_flooding/processed_data/infrastructure/africa_road_edges_network.geoparquet \
            --output_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_19495/road_edges_network.geoparquet
    """
    
    crs = "EPSG:4326"
    HAZ = gpd.read_file(haz_path)
    HAZ = HAZ[HAZ["T500_ID"].astype(str) == haz_id] 

    # upload input data
    asset = gpd.read_parquet(asset_path)
   
    # ensure they have same crs
    
    asset = asset.to_crs(crs)
    HAZ = HAZ.to_crs(crs)
  
    
    # clip to HAZ
    asset_clipped = gpd.clip(asset, HAZ)

    
    # save outputs
    
    if len(asset_clipped) > 0:
        asset_clipped.to_parquet(output_path)
        logging.info("Clipped asset data saved.")
    else:
        logging.info("No intersection found; no file created.")

    logging.info("Done.")



if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s",
        level=logging.INFO
    )

    main()