
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
    "--defended_areas_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    help="Path to defended areas Geoparquet",
)
@click.option(
    "--output_path",
    required=True,
    type=click.Path(exists=False, dir_okay=False, file_okay=True, writable=True),
    help="Path to clipped output file",
)

def main(haz_id, haz_path , defended_areas_path , output_path):
    """
    Example usage:

        python workflow/scripts/clip_to_HAZ.py \
            --haz_id 500_13_19495 \
            --haz_path ~/Desktop/DataFolders/JBA_flooding/processed_data/basins/haz_500.gpkg \
            --defended_areas_path ~/Desktop/DataFolders/JBA_flooding/processed_data/defended_areas/defended_areas.gpkg \
            --output_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_19495/defended_areas_prova.gpkg
    """
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    crs = "EPSG:4326"
    HAZ = gpd.read_file(haz_path)
    HAZ = HAZ[HAZ["T500_ID"].astype(str) == haz_id] 

    # upload input data
    
    defended = gpd.read_file(defended_areas_path)
    
    # ensure they have same crs
    
    HAZ = HAZ.to_crs(crs)
    defended = defended.to_crs(crs) 
    
    # clip to HAZ

    defended_clipped = gpd.clip(defended, HAZ)
    
    # save outputs
    
    if len(defended_clipped) > 0:
        defended_clipped.to_parquet(output_path)
    
        logging.info("Clipped defended areas saved.")
    
    else:
        logging.info("No intersection found; saving empty output.")
        empty_gdf = gpd.GeoDataFrame([], geometry=[], crs=crs)
        empty_gdf.to_parquet(output_path)
        
        logging.info("Done.")



if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s",
        level=logging.INFO
    )

    main()