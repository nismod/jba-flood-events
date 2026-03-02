import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as riox
from shapely.geometry import Polygon


def main(input, output):

    HAZ = gpd.read_file(input.gpkg)

    haz_id = snakemake.wildcards.HAZ

    HAZ = HAZ[HAZ["T500_ID"] == haz_id]   

    hazard = riox.open_rasterio(input.rp_tiff)
    asset = gpd.read_parquet(input.geoparquet)
    defended = gpd.read_file(input.defended_areas)
    # ensure they have same crs
    hazard = hazard.rio.reproject(HAZ.crs)
    asset = asset.to_crs(HAZ.crs)
    

    rp_clipped = hazard.rio.clip(
        HAZ.geometry,
        HAZ.crs,
        drop=True
    )
    asset_clipped = gpd.clip(asset, HAZ)
    defended_clipped = gpd.clip(defended, HAZ)

    rp_clipped.rio.to_raster(output.rp_tiff)
    asset_clipped.to_parquet(output.exposed)
    defended_clipped.to_file(output.defended, driver="GPKG")


    logging.info("Done.")


if __name__ == "__main__":

    logging.basicConfig(
        filename=snakemake.log.file,
        format="%(asctime)s %(process)d %(filename)s %(message)s",
        level=logging.INFO
    )

    input = snakemake.input
    output = snakemake.output

    result = main(input, output)