import click
import logging
import os
import re
import sys
import warnings

from glob import glob
from functools import partial
from pathlib import Path
from rasterio.plot import show

import pandas as pd
import geopandas as gpd
import rasterio
import rioxarray
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import openpyxl
from scipy.stats.mstats import gmean
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import snail.intersection as snint
import shapely

import snakemake  


@click.command()
@click.version_option("1.0")

@click.option(
    "--undefended_path",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True),
    help="Path to undefended depth tiffs",
)
@click.option(
    "--sop_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    help="Path to defended areas depth geoparquet",
)
@click.option(
    "--output_path",
    required=True,
    type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True),
    help="Path to defended depths tiffs",
)



def main(undefended_path, sop_path, output_path):
    """
    Example usage:

        python workflow/scripts/defended_depths.py \
            --undefended_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/ObsEvents/Undefended/road_edges_network/\
            --sop_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/defroad_edges_network/split_def.geoparquet\
            --output_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/ObsEvents/Defended/road_edges_network/
    """
    os.makedirs(output_path, exist_ok=True)

    # upload input data

    undefended_files = sorted(glob(os.path.join(undefended_path, "event=*/*.parquet")))

    # Load SoP depths 
    sop = pd.read_parquet(sop_path)  # columns: raster_i, raster_j, depth
    sop.rename(columns={"depth": "sop_depth"}, inplace=True)
    


    for parquet_path in undefended_files:
        # Read the event parquet file
        df = pd.read_parquet(parquet_path)

        # Merge with SoP on raster_i, raster_j
        merged = df.merge(
            sop[["raster_i", "raster_j", "sop_depth"]],
            on=["raster_i", "raster_j"],
            how="left",
        )

        # depth = tif_depth - sop_depth where SoP exists, else tif_depth
        merged["depth"] = np.where(
            merged["sop_depth"].notna(),
            merged["depth"] - merged["sop_depth"],
            merged["depth"],
        )
        merged["depth"] = merged["depth"].clip(lower=0)  # Ensure no negative depths

        merged = merged.drop(columns=["sop_depth"])
        
        event_folder = os.path.basename(os.path.dirname(parquet_path))  
        filename = os.path.basename(parquet_path)                        
        out_path = os.path.join(output_path, event_folder, filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        merged.to_parquet(out_path, index=False)
    
    sentinel = Path(output_path) / ".done"
    sentinel.touch()

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s",
        level=logging.INFO
    )

    main()