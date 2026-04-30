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
    "--rp_path",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True),
    help="Path to RP TIFF",
)
@click.option(
    "--defended_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    help="Path to defended areas geoparquet",
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
    help="Path to defended areas depths output file",
)



def main(rp_path, defended_path, asset_path, output_path):
    """
    Example usage:

        python workflow/scripts/defended_areas_to_grid.py \
            --rp_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/ \
            --defended_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/defended_areas.geoparquet\
            --asset_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/split_road_edges_network.geoparquet \
            --output_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/split_defended_areas/road_edges_network/split_defended_areas.geoparquet
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logging.info("Reading raster and defended areas data...")
    

    # upload input data

    crs = "EPSG:4326"
    haz_id = rp_path.rstrip("/").split("/")[-1]  # Extract HAZ ID from filename, assuming it's the filename without extension
   

    vector = gpd.read_parquet(defended_path)
    vector = vector.to_crs(crs)
    if len(vector) == 0:
        logging.info("No defended areas in this HAZ, writing empty file.")
        vector.to_parquet(output_path)
        return
    vector['geometry'] = vector['geometry'].apply(
            lambda g: shapely.force_2d(g)  # ensure 2D
        )
    vector = vector.explode(index_parts=False)  # MultiLineString -> LineString etc.
    vector = vector.reset_index(drop=True)
    vector["T500_ID"] = haz_id
    print("vector types:", vector.geometry.geom_type.value_counts().to_dict())

    # Check for invalid/null geometries
    print("Total rows:", len(vector))
    print("Null geometries:", vector.geometry.isna().sum())
    print("Invalid geometries:", (~vector.geometry.is_valid).sum())
    print(vector.total_bounds)
    
    
    rp_files = sorted(glob(os.path.join(rp_path, "flrf_ud_Q*.tif")))    
    river_exposure_points, profile = read_rp_maps_to_points(rp_files) # reads all RP TIFFs in the directory and returns a GeoDataFrame of points with RP values and the raster profile for output
    
    with rasterio.open(rp_files[1]) as src:
        bounds = src.bounds
        
    grid, window = grid_from_window(rp_files[1], bounds)
    
    river_exposure_points = snint.apply_indices(
            river_exposure_points, grid, index_i="raster_i", index_j="raster_j"
        )

    river_exposure_points["T500_ID"] = haz_id

    logging.info("Splitting polygons...")
    vector = vector.reset_index(drop=True)
    vector_splits = snint.split_polygons(vector, grid)
    logging.info("Split %d polygons into %d pieces", len(vector), len(vector_splits))
    
    logging.info("Finding indices...")
    vector_splits = snint.apply_indices(
        vector_splits, grid, index_i="raster_i", index_j="raster_j"
    )
    
    vector_splits.rename(columns={"JBA_SoP": "rp"}, inplace=True)
    
    asset = gpd.read_parquet(asset_path)
    asset["asset"] = asset_path.split("/")[-1].replace(".geoparquet", "").replace("split_", "")
    
    output_dir = os.path.dirname(output_path)




    interpolate_event_exposure(
        vector_splits,
        river_exposure_points,
        hazard_prefix="FLRF",
        asset=asset,
        output_dir=output_dir,
    )

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

    event_zones["def_id"] = range(len(event_zones))
    event_ids = sorted(event_zones.reset_index()["def_id"].unique())

    
    event_depths(
        event_zones=event_zones,
        exposure_points=exposure_points,
        hazard_prefix=hazard_prefix,
        asset=asset,
        output_dir=output_dir,
    )
   


def event_depths(event_zones, exposure_points, hazard_prefix, asset, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "split_def.geoparquet")
    
    event_points = exposure_points.merge(
        event_zones[["raster_i", "raster_j", "rp", "bin_index", "rp_l", "rp_u", "rp_factor"]],
        on=["raster_i", "raster_j"],
        how="inner"
    )
    print(event_points)

    if len(event_points) == 0:
        logging.warning("No overlapping raster cells between event zones and exposure points, writing empty file.")
        event_points.to_parquet(output_path, index=False)
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
        event_points = event_points.reset_index()[["depth", "raster_i", "raster_j"]]

        
        event_points = event_points[event_points.depth > 0]
        event_points["hazard"] = hazard_prefix

        if len(asset) == 0:
            logging.warning(f"Asset file is empty, skipping: {asset}")
            os.makedirs(output_dir, exist_ok=True)
            event_points.iloc[0:0].to_parquet(os.path.join(output_dir, "split_def.geoparquet"), index=False)
            return

        event_points = event_points.merge(asset[['raster_i', 'raster_j']], on=['raster_i', 'raster_j'], how='inner')

        event_points.to_parquet(os.path.join(output_dir, "split_def.geoparquet"), index=False)

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