
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
    "--rp_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    help="Path to RP TIFF",
)
@click.option(
    "--output_path",
    required=True,
    type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True),
    help="Path to clipped output directory",
)
def main(haz_id, haz_path, rp_path, output_path):
    """
    Example usage:

        python workflow/scripts/clip_rp_to_HAZ.py \
            --haz_id 500_13_19495 \
            --haz_path ~/Desktop/DataFolders/JBA_flooding/processed_data/basins/haz_500.gpkg \
            --rp_path ~/Desktop/DataFolders/JBA_flooding/processed_data/hazards/flrf_ud_Q500.tif \
            --output_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_19495/
    """
    os.makedirs(output_path, exist_ok=True)
    crs = "EPSG:4326"
    HAZ = gpd.read_file(haz_path)
    HAZ = HAZ[HAZ["T500_ID"].astype(str) == haz_id] 

    # upload input data
    # hazard = riox.open_rasterio(input.rp_tiff)
    # asset = gpd.read_parquet(input.asset_geoparquet)
    hazard = riox.open_rasterio(rp_path)
    # defended = gpd.read_file(defended_areas_path)
    
    # ensure they have same crs
    hazard = hazard.rio.reproject(crs)
    # asset = asset.to_crs(crs)
    HAZ = HAZ.to_crs(crs)
    #defended = defended.to_crs(crs) 
    
    # clip hazard, asset and defended areas to HAZ

    rp_clipped = hazard.rio.clip(
        HAZ.geometry,
        HAZ.crs,
        drop=True
    )
    # asset_clipped = gpd.clip(asset, HAZ)
    #defended_clipped = gpd.clip(defended, HAZ)
    
    # save outputs
    
    # rp_clipped.rio.to_raster(output.rp_clipped)
    # asset_clipped.to_parquet(output.exposed_clipped)
    if len(rp_clipped) > 0:
        rp_clipped.rio.to_raster(output_path+"flrf_ud_Q500.tif")
        logging.info("Clipped RP raster saved.")
    else:
        logging.info("No intersection found; no file created.")

    logging.info("Done.")



if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s",
        level=logging.INFO
    )

    main()