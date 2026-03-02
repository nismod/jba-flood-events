import os
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as riox
from shapely.geometry import Polygon


def main(input, output, params):

    HAZ = gpd.read_file(input.gpkg)

    # Filter to the specific basin requested by Snakemake wildcard
    haz_id = snakemake.wildcards.HAZ
    HAZ = HAZ[HAZ["T500_ID"] == haz_id]   # <-- adjust column name if needed

    hazard = riox.open_rasterio(input.tiff)

    if HAZ.crs != hazard.rio.crs:
        HAZ = HAZ.to_crs(hazard.rio.crs)

    clipped = hazard.rio.clip(
        HAZ.geometry,
        HAZ.crs,
        drop=True
    )

    clipped.rio.to_raster(output.tiff)

    logging.info("Done.")


if __name__ == "__main__":

    logging.basicConfig(
        filename=snakemake.log.file,
        format="%(asctime)s %(process)d %(filename)s %(message)s",
        level=logging.INFO
    )

    input = snakemake.input
    output = snakemake.output
    params = snakemake.params

    result = main(input, output, params)