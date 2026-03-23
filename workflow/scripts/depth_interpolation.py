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

import snakemake  # Add this import at the top if not already present
    

RPS = np.array([1e-3, 2, 20, 50, 100, 200, 500, 1500, 1e6])


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
    rp = re.search(r"Q(\d+)_", fname).group(1)
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


def link_event_op_haz(events, haz_op):
    # Link event OPs to HAZ (drop OPs which are not linked)
    # op.id, event.id, rp, T500_ID
    events = events.reset_index().merge(haz_op, on="op.id").dropna()

    # Take the geometric mean of Event/OP return periods if multiple OPs per HAZ
    # (T500_ID, event.id) rp
    return events.groupby(["event.id", "T500_ID"]).agg({"rp": gmean})

def interpolate_rp_factor(df):
    return (np.log(df.rp) - np.log(df.rp_l)) / (np.log(df.rp_u) - np.log(df.rp_l))

def interpolate_event_exposure(event_zones, exposure_points, hazard_prefix, output_path, profile):
    event_zones.loc[event_zones.rp >= 1500, "rp"] = 1500
    bin_index = np.searchsorted(RPS, event_zones.rp, side="left")
    event_zones["bin_index"] = bin_index
    event_zones["rp_l"] = RPS[bin_index - 1]
    event_zones["rp_u"] = RPS[bin_index]
    event_zones["rp_factor"] = interpolate_rp_factor(event_zones)
    
    # Process only the single event
    event_id = event_zones.index.get_level_values('event.id').unique()[0]
    event_depths(event_id, event_zones, exposure_points, hazard_prefix, output_path, profile)

def interpolate_depth_df(df):
    depth = df.depth_l + ((df.depth_u - df.depth_l) * df.rp_factor)

    return depth

def event_depths(event_id, event_zones, exposure_points, hazard_prefix, output_path, profile):
    event_haz = event_zones.loc[event_id].reset_index()
    event_points = exposure_points.loc[event_haz.T500_ID].join(event_haz.set_index("T500_ID"))
    event_points.bin_index = event_points.bin_index.astype(np.int32)
    
    if len(event_points):
        depths = [0, event_points.rp2, event_points.rp20, event_points.rp50, event_points.rp100, event_points.rp200, event_points.rp500, event_points.rp1500]
        event_points["depth_l"] = np.choose(event_points.bin_index - 1, depths)
        event_points["depth_u"] = np.choose(event_points.bin_index, depths)
        event_points["depth"] = interpolate_depth_df(event_points)
        event_points.loc[event_points.rp <= 2, "depth"] = 0
        
        # Create the depth raster
        depth_array = np.full((profile['height'], profile['width']), 0, dtype=np.float32)
        depth_array.flat[event_points.index] = event_points["depth"]  # cell_index is the index
        
        # Write the TIFF
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(depth_array, 1)

def link_haz_op(haz, ops):
    # Link OPs to HAZs
    haz_within = haz.sjoin(ops, predicate="contains", how="right")[["T500_ID", "op.id"]]

    # some HAZ do not contain an OP
    haz_remaining = haz[~haz.T500_ID.isin(haz_within.T500_ID.unique())]
    haz_nearest = haz_remaining.sjoin_nearest(ops, how="left")[["T500_ID", "op.id"]]
    return pd.concat([haz_within, haz_nearest])


def main(input, output):
    # Access wildcards here
    HAZ = snakemake.wildcards.HAZ
    EVENT = snakemake.wildcards.EVENT
    
    river_ops = latlon_to_gdf(
        pd.read_csv(input.csv2, lat_column="op.lat", lon_column="op.lon")[["op.id", "geometry"]]
    )
    hydrological_accumulation_zones = gpd.read_file(input.gpkg)[["T500_ID", "geometry"]]
    river_ops = link_haz_op(hydrological_accumulation_zones, river_ops)
    
    # Glob the RP TIFFs from the directory
    rp_files = sorted(glob(os.path.join(input.rp, "flrf_ud_Q*.tif")))
    river_rp_points, profile = read_rp_maps_to_points(rp_files)
    
    event_set = pd.read_csv(
        input.csv1, usecols=["event.id", "op.id", "rp"]
    ).set_index(["op.id", "event.id"])
    
    # Filter to the specific EVENT
    event_set = event_set.xs(EVENT, level='event.id')
    
    river_events = link_event_op_haz(event_set, river_ops)
    
    # Filter to the specific HAZ (T500_ID)
    river_events = river_events.xs(HAZ, level='T500_ID')
    
    interpolate_event_exposure(
        river_events,
        river_rp_points,
        hazard_prefix="FLRF",
        output_path=output.depth,  
        profile=profile
    )

if __name__ == "__main__":

    logging.basicConfig(
        filename=snakemake.log[0],
        format="%(asctime)s %(process)d %(filename)s %(message)s",
        level=logging.INFO
    )

    input = snakemake.input
    output = snakemake.output
   
    
    result = main(input, output)