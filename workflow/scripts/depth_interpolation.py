from posixpath import dirname
from unittest.mock import sentinel

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

import snakemake  

@click.command()
@click.version_option("1.0")

@click.option(
    "--op_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    help="Path to OP csv",
)
@click.option(
    "--info_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    help="Path to RiverOpInfo csv",
)

@click.option(
    "--rp_path",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True),
    help="Path to folder containing RP TIFFs",
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
    type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True),
    help="Path to interpolated event tiff file",
)


def main(haz_path, rp_path, op_path, info_path, asset_path, output_path):
    
    """
    Example usage:

        python workflow/scripts/depth_interpolation.py \
            --haz_path ~/Desktop/DataFolders/JBA_flooding/processed_data/basins/haz_500.gpkg \
            --rp_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/ \
            --info_path ~/Desktop/DataFolders/JBA_flooding/processed_data/events/RiverOpInfo.gpkg \
            --op_path ~/Desktop/DataFolders/JBA_flooding/processed_data/events/ObsEventRp.csv \
            --asset_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/split_road_edges_network.geoparquet \
            --output_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/ObsEvents/Undefended/road_edges
    """
    
    
    os.makedirs(output_path, exist_ok=True)

    crs = "EPSG:4326"
    haz_id = rp_path.rstrip("/").split("/")[-1]  # Extract HAZ ID from filename, assuming it's the filename without extension
    hydrological_accumulation_zones = gpd.read_file(haz_path)[["T500_ID", "geometry"]].to_crs(crs)
    hydrological_accumulation_zones = hydrological_accumulation_zones[hydrological_accumulation_zones["T500_ID"] == haz_id]

    
    river_ops = gpd.read_file(info_path)[["op.id", "geometry"]].to_crs(crs) 
    
    river_haz_op = link_haz_op(hydrological_accumulation_zones, river_ops)
    river_haz_op = river_haz_op.dropna()

    # Glob the RP TIFFs from the directory
    
    rp_files = sorted(glob(os.path.join(rp_path, "flrf_ud_Q*.tif")))
    river_rp_points, profile = read_rp_maps_to_points(rp_files) # reads all RP TIFFs in the directory and returns a GeoDataFrame of points with RP values and the raster profile for output
    
    with rasterio.open(rp_files[1]) as src:
        bounds = src.bounds
        
    grid, window = grid_from_window(rp_files[1], bounds)
    
    river_exposure_points = link_haz_ep(
            hydrological_accumulation_zones, river_rp_points
        )
    river_exposure_points = snint.apply_indices(
            river_exposure_points, grid, index_i="raster_i", index_j="raster_j"
        )
    # river_exposure_points.to_parquet(os.path.join((Path(output_path).parent), "exposure_points_prova.parquet"), index=True)

    event_set = pd.read_csv(
        op_path, usecols=["event.id", "op.id", "rp"]
    ).set_index(["op.id", "event.id"])
  
       
    river_events = link_event_op_haz(event_set, river_haz_op)
    asset = gpd.read_parquet(asset_path)
    asset["asset"] = asset_path.split("/")[-1].replace(".geoparquet", "").replace("split_", "")
    
    
    
    interpolate_event_exposure(
        river_events,
        river_exposure_points,
        hazard_prefix="FLRF",
        asset=asset,
        output_dir=output_path,
    )

    sentinel = Path(output_path) / ".done"
    sentinel.touch()



def link_haz_op(haz, ops):
    haz_proj = haz.to_crs("EPSG:3857")
    ops_proj = ops.to_crs("EPSG:3857")
    
    haz_within = haz_proj.sjoin(ops_proj, predicate="contains", how="right")[["T500_ID", "op.id"]]
    
    haz_remaining = haz_proj[~haz_proj.T500_ID.isin(haz_within.T500_ID.unique())]
    haz_nearest = haz_remaining.sjoin_nearest(ops_proj, how="left")[["T500_ID", "op.id"]]
    
    return pd.concat([haz_within, haz_nearest])

def latlon_to_gdf(df, lat_column="lat", lon_column="lon"):
    geometry = gpd.points_from_xy(df[lon_column], df[lat_column])
    return gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)

def get_xy_df(fname):
    """Read a raster file, convert to dataframe retaining only x,y coordinate
    values  and 0..n index
    """
    xy = (
        rioxarray.open_rasterio(fname)
        .to_dataframe("data")
        .reset_index()
        .drop(columns=["band", "spatial_ref", "data"])
    )
    return xy

def read_rp_map(fname):
    """Read flood maps, use all cells with any depth > 0 as potential exposure
    points
    """
    rp = re.search(r"Q(\d+)", fname).group(1)
    colname = f"rp{rp}"
    with rasterio.open(fname) as dataset:
        data = dataset.read(1)
        data[data == dataset.nodata] = np.nan
        df = pd.DataFrame({colname: data.flatten()})

    df = df[df[colname] > 0].dropna()
    return df

def read_rp_maps_to_points(pattern):
    # Define as an iter to use each file once
    rp_maps = iter(sorted(glob(pattern)))

    # Read first
    rp_points = read_rp_map(next(rp_maps))
    # Read the rest
    for fname in rp_maps:
        df = read_rp_map(fname)
        rp_points = rp_points.join(df, how="outer")

    # Fill NA with zeros
    rp_points = rp_points.fillna(0)
    rp_points["rp2"] = 0
    xy = get_xy_df(fname)
    rp_points = rp_points.join(xy)

    # name index
    rp_points.index.rename("cell_index", inplace=True)

    # Convert to GeoDataFrame
    return latlon_to_gdf(rp_points, lat_column="y", lon_column="x").drop(
        columns=["y", "x"]
    )

def read_rp_map(fname):
    """Read flood maps, use all cells with any depth > 0 as potential exposure
    points
    """
    rp = re.search(r"Q(\d+)", fname).group(1)
    colname = f"rp{rp}"
    with rasterio.open(fname) as dataset:
        data = dataset.read(1)
        data[data == dataset.nodata] = np.nan
        df = pd.DataFrame({colname: data.flatten()})

    df = df[df[colname] > 0].dropna()
    return df

def read_rp_maps_to_points(rp_files):
    # rp_files is now a list of paths
    rp_maps = iter(sorted(rp_files))  
    
    rp_points = read_rp_map(next(rp_maps))
    for fname in rp_maps:
        df = read_rp_map(fname)
        rp_points = rp_points.join(df, how="outer")
    
    rp_points = rp_points.fillna(0)
    rp_points["rp2"] = 0
    xy = get_xy_df(fname)  # fname is the last file, assume all have same grid
    rp_points = rp_points.join(xy)
    rp_points.index.rename("cell_index", inplace=True)
    
    # Get profile from the last file (assume all RP TIFFs have the same profile)
    with rasterio.open(fname) as src:
        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, count=1, nodata=0)  # Update for depth output
    
    return latlon_to_gdf(rp_points, lat_column="y", lon_column="x").drop(columns=["y", "x"]), profile

def link_haz_ep(haz, eps):
    return (
        eps.sjoin(haz, predicate="within", how="left"
        )
        .reset_index()
        .drop(columns="index_right")
        .set_index("T500_ID")
    )

def link_event_op_haz(events, haz_op):
    # Link event OPs to HAZ (drop OPs which are not linked)
    # op.id, event.id, rp, T500_ID
    events = events.reset_index().merge(haz_op, on="op.id").dropna()

    # Take the geometric mean of Event/OP return periods if multiple OPs per HAZ
    # (T500_ID, event.id) rp
    return events.groupby(["event.id", "T500_ID"]).agg({"rp": gmean})

def interpolate_rp_factor(df):
    return (np.log(df.rp) - np.log(df.rp_l)) / (np.log(df.rp_u) - np.log(df.rp_l))


def interpolate_depth_df(df):
    depth = df.depth_l + ((df.depth_u - df.depth_l) * df.rp_factor)

    return depth

def interpolate_event_exposure(event_zones, exposure_points, hazard_prefix, asset, output_dir):
    
    RPS = np.array([1e-3, 2, 20, 50, 100, 200, 500, 1500, 1e6])
    # Cap at max RP 1500
    event_zones.loc[event_zones.rp >= 1500, "rp"] = 1500

    bin_index = np.searchsorted(RPS, event_zones.rp, side="left")
    event_zones["bin_index"] = bin_index
    event_zones["rp_l"] = RPS[bin_index - 1]
    event_zones["rp_u"] = RPS[bin_index]
    event_zones["rp_factor"] = interpolate_rp_factor(event_zones)
    # event_zones is now a dataframe with:
    # (T500_ID, event.id) rp, bin_index, rp_l, rp_u, rp_factor

    event_ids = sorted(event_zones.reset_index()["event.id"].unique())
    event_depths_partial = partial(
        event_depths,
        event_zones=event_zones,
        exposure_points=exposure_points,
        hazard_prefix=hazard_prefix,
        asset=asset,
        output_dir=output_dir,
    )
    process_map(
        event_depths_partial,
        event_ids,
        chunksize=32,
        max_workers=int(os.cpu_count() / 2),
    )


def event_depths(event_id, event_zones, exposure_points, hazard_prefix, asset, output_dir):
    
    
    # Each HAZ in this event, with RP values
    event_haz = event_zones.loc[event_id].reset_index()
    # All points for this event, joined with RP values via HAZ
    event_points = exposure_points.loc[event_haz.T500_ID].join(
        event_haz.set_index("T500_ID")
    )
    if len(event_points) == 0:
        logging.warning("No overlapping raster cells between event zones and exposure points, writing empty file.")
        event_points.to_parquet(output_dir, index=False)
        return
    event_points.bin_index = event_points.bin_index.astype(np.int32)

    if len(event_points):
        depths = [
            0,
            event_points.rp2,
            event_points.rp20,
            event_points.rp50,
            event_points.rp100,
            event_points.rp200,
            event_points.rp500,
            event_points.rp1500,
        ]
        event_points["depth_l"] = np.choose(event_points.bin_index - 1, depths)
        event_points["depth_u"] = np.choose(event_points.bin_index, depths)
        event_points["depth"] = interpolate_depth_df(event_points)
        # Any RP < 2 gets zero depth
        event_points.loc[event_points.rp <= 2, "depth"] = 0

        # Output cells
        # T500_ID, depth, cell_index, event
        event_points = event_points.reset_index()[["T500_ID", "depth", "cell_index", "raster_i", "raster_j"]]

        
        event_points = event_points[event_points.depth > 0]
        event_points["event"] = event_id
        event_points["hazard"] = hazard_prefix

        if len(asset) == 0:
            logging.warning(f"Asset file is empty, skipping: {asset}")
            os.makedirs(output_dir, exist_ok=True)  # create empty output dir so Snakemake is satisfied
            return
        
        event_points = event_points.merge(asset[['raster_i', 'raster_j','id']], on=['raster_i', 'raster_j'], how='inner')
        
        event_points.to_parquet(
            output_dir, partition_cols=["event"], index=False
        )


def link_haz_op(haz, ops):
    haz_proj = haz.to_crs("EPSG:3857")
    ops_proj = ops.to_crs("EPSG:3857")
    
    haz_within = haz_proj.sjoin(ops_proj, predicate="contains", how="right")[["T500_ID", "op.id"]]
    
    haz_remaining = haz_proj[~haz_proj.T500_ID.isin(haz_within.T500_ID.unique())]
    haz_nearest = haz_remaining.sjoin_nearest(ops_proj, how="left")[["T500_ID", "op.id"]]
    
    return pd.concat([haz_within, haz_nearest])


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

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s",
        level=logging.INFO
    )

    main()