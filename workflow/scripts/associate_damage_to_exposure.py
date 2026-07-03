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
import snail.damages as sndam

import snakemake  

@click.command()
@click.version_option("1.0")

@click.option(
    "--depth_path",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True),
    help="Path to depth file folder",
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
    help="Path to damage geoparquet file",
)


def main(depth_path, asset_path, cost_path, curves_path, output_path):
    
    """
    Example usage:

        python workflow/scripts/associate_damage_to_exposure.py \
            --depth_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/ObsEvents/Undefended/road_edges/ \
            --asset_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/split_road_edges_network.geoparquet \
            --cost_path ~/Desktop/DataFolders/JBA_flooding/processed_data/costs/road/road_edges_costs.csv \
            --curves_path ~/Desktop/DataFolders/JBA_flooding/processed_data/curves/road/\
            --output_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/ObsCosts/Undefended/road_edges/
    """
    
    
    os.makedirs(output_path, exist_ok=True)

    # upload input data

    depth_files = sorted(glob(os.path.join(depth_path, "event=*/*.parquet")))
    asset_type = os.path.basename(asset_path).split("_")[1]  # e.g. "road"
    
    # Load depths 
    depth = pd.read_parquet(depth_path)      
    asset = gpd.read_parquet(asset_path)  
    cost = pd.read_csv(cost_path)
    curves = glob(os.path.join(curves_path, "*.csv"))
    
    for parquet_path in depth_files:
        # Derive the event folder name, e.g. "event=001"
        event_folder = os.path.basename(os.path.dirname(parquet_path))  # "event=001"

        # Read the per-event depth data
        depth_event = pd.read_parquet(parquet_path)  # expects columns: id, depth_m
        depth_event = depth_event.rename(columns={"depth": "depth_m"})

        # Merge depth_m onto asset using the shared "id" column
        asset_event = asset.merge(
            depth_event[["id", "depth_m"]],
            on="id",
            how="left",          # keep all assets; unmatched get NaN depth
        )
        
        if asset_type == "road":

            asset_event["cost_usd_per_m"] = np.where(
                asset_event["bridge"] == True,
                cost.loc[cost["asset_type"] == "bridge", "mean_cost_usd_per_m"].values[0],
                np.where(
                    asset_event["paved"] == True,
                    cost.loc[cost["asset_type"] == "paved", "mean_cost_usd_per_m"].values[0],
                    cost.loc[cost["asset_type"] == "unpaved", "mean_cost_usd_per_m"].values[0]
                )
            )
                        
            
            damage_curve_bridge = sndam.from_csv(str, curves["bridge"], intensity_col="flood_depth", damage_col="damage_fraction_mean", comment="#", **kwargs,)
            damage_curve_paved = sndam.from_csv(str, curves["road_paved"], intensity_col="flood_depth", damage_col="damage_fraction_mean", comment="#", **kwargs,)
            damage_curve_unpaved = sndam.from_csv(str, curves["road_unpaved"], intensity_col="flood_depth", damage_col="damage_fraction_mean", comment="#", **kwargs,)

            paved_depths = asset_event.loc[asset_event.paved==True & asset_event.bridge==False, "depth_m"]
            paved_damage = damage_curve_paved.damage_fraction(paved_depths)
            asset_event.loc[asset_event.paved==True, "proportion_damaged"] = paved_damage

            unpaved_depths = asset_event.loc[asset_event.paved==False & asset_event.bridge==False, "depth_m"]
            unpaved_damage = damage_curve_unpaved.damage_fraction(unpaved_depths)
            asset_event.loc[asset_event.paved==False, "proportion_damaged"] = unpaved_damage

            bridge_depths = asset_event.loc[asset_event.bridge==True, "depth_m"]
            bridge_damage = damage_curve_bridge.damage_fraction(bridge_depths)
            asset_event.loc[asset_event.bridge==True, "proportion_damaged"] = bridge_damage

            asset_event["damage_usd"] = asset_event["length_m"] * asset_event["cost_usd_per_m"] * asset_event["proportion_damaged"]

        elif asset_type == "railway":

            is_bridge = asset_event["structure"].isin(["bridge", "viaduct"])
            is_mgr    = asset_event["gauge"] <= 1000
            is_sgr    = asset_event["gauge"] > 1000

            def cost_val(asset_type_str):
                return cost.loc[cost["asset_type"] == asset_type_str, "mean_cost_usd_per_m"].values[0]

            asset_event["cost_usd_per_m"] = np.select(
                [
                    # Metre gauge (mgr)
                    is_mgr & is_bridge  & (asset_event["status"] == "open"),
                    is_mgr & is_bridge  & (asset_event["status"] == "disused"),
                    is_mgr & ~is_bridge & (asset_event["status"] == "open"),
                    is_mgr & ~is_bridge & (asset_event["status"] == "disused"),
                    # Standard gauge (sgr)
                    is_sgr & is_bridge  & (asset_event["status"] == "open"),
                    is_sgr & ~is_bridge & (asset_event["status"] == "open"),
                    is_sgr & ~is_bridge & (asset_event["status"] == "construction"),
                    is_sgr & ~is_bridge & (asset_event["status"] == "planned"),
                    is_sgr & ~is_bridge & (asset_event["status"] == "proposed"),
                ],
                [
                    cost_val("mgr_bridge_open"),
                    cost_val("mgr_bridge_disused"),
                    cost_val("mgr_track_open"),
                    cost_val("mgr_track_disused"),
                    cost_val("mgr_bridge_open"),   # sgr bridge open → mgr_bridge_open
                    cost_val("sgr_track_open"),
                    cost_val("sgr_track_construction"),
                    cost_val("sgr_track_planned"),
                    cost_val("sgr_track_proposed"),
                ],
                default=0  # all other statuses
            )
            

            damage_curve_mgr_bridge       = sndam.from_csv(curves["mgr_bridge_open"],        intensity_col="flood_depth", damage_col="damage_fraction_mean", comment="#")
            damage_curve_mgr_track        = sndam.from_csv(curves["mgr_track_open"],         intensity_col="flood_depth", damage_col="damage_fraction_mean", comment="#")
            damage_curve_sgr_construction = sndam.from_csv(curves["sgr_track_construction"], intensity_col="flood_depth", damage_col="damage_fraction_mean", comment="#")
            damage_curve_sgr_open         = sndam.from_csv(curves["sgr_track_open"],         intensity_col="flood_depth", damage_col="damage_fraction_mean", comment="#")
            damage_curve_sgr_planned      = sndam.from_csv(curves["sgr_track_planned"],      intensity_col="flood_depth", damage_col="damage_fraction_mean", comment="#")
            damage_curve_sgr_proposed     = sndam.from_csv(curves["sgr_track_proposed"],     intensity_col="flood_depth", damage_col="damage_fraction_mean", comment="#")

            asset_event["proportion_damaged"] = 0.0

            # Metre gauge bridge
            mgr_bridge_depths = asset_event.loc[is_mgr & is_bridge & (asset_event["status"] == "open"), "depth_m"]
            asset_event.loc[is_mgr & is_bridge & (asset_event["status"] == "open"), "proportion_damaged"] = damage_curve_mgr_bridge.damage_fraction(mgr_bridge_depths)

            # Metre gauge track
            mgr_track_depths = asset_event.loc[is_mgr & ~is_bridge & (asset_event["status"] == "open"), "depth_m"]
            asset_event.loc[is_mgr & ~is_bridge & (asset_event["status"] == "open"), "proportion_damaged"] = damage_curve_mgr_track.damage_fraction(mgr_track_depths)

            # Standard gauge bridge (uses mgr_bridge_open curve)
            sgr_bridge_depths = asset_event.loc[is_sgr & is_bridge & (asset_event["status"] == "open"), "depth_m"]
            asset_event.loc[is_sgr & is_bridge & (asset_event["status"] == "open"), "proportion_damaged"] = damage_curve_mgr_bridge.damage_fraction(sgr_bridge_depths)

            # Standard gauge track
            sgr_open_depths = asset_event.loc[is_sgr & ~is_bridge & (asset_event["status"] == "open"), "depth_m"]
            asset_event.loc[is_sgr & ~is_bridge & (asset_event["status"] == "open"), "proportion_damaged"] = damage_curve_sgr_open.damage_fraction(sgr_open_depths)

            sgr_construction_depths = asset_event.loc[is_sgr & ~is_bridge & (asset_event["status"] == "construction"), "depth_m"]
            asset_event.loc[is_sgr & ~is_bridge & (asset_event["status"] == "construction"), "proportion_damaged"] = damage_curve_sgr_construction.damage_fraction(sgr_construction_depths)

            sgr_planned_depths = asset_event.loc[is_sgr & ~is_bridge & (asset_event["status"] == "planned"), "depth_m"]
            asset_event.loc[is_sgr & ~is_bridge & (asset_event["status"] == "planned"), "proportion_damaged"] = damage_curve_sgr_planned.damage_fraction(sgr_planned_depths)

            sgr_proposed_depths = asset_event.loc[is_sgr & ~is_bridge & (asset_event["status"] == "proposed"), "depth_m"]
            asset_event.loc[is_sgr & ~is_bridge & (asset_event["status"] == "proposed"), "proportion_damaged"] = damage_curve_sgr_proposed.damage_fraction(sgr_proposed_depths)

            asset_event["damage_usd"] = (
                asset_event["length_m"] * asset_event["cost_usd_per_m"] * asset_event["proportion_damaged"]
            )

        elif asset_type == "iww":
            
                asset_event["cost_usd_per_m"] = cost.loc[cost["asset_type"] == "general cargo", "mean_cost_usd_per_sqm"].values[0]
                asset_event["area_sqm"] = asset_event.to_crs("EPSG:102022").geometry.area
                damage_curve = sndam.from_csv(curves["general cargo"], intensity_col="flood_depth", damage_col="damage_fraction_mean", comment="#")
                asset_event["proportion_damaged"] = damage_curve.damage_fraction(asset_event["depth_m"])
                asset_event["damage_usd"] = asset_event["area_sqm"] * asset_event["cost_usd_per_sqm"] * asset_event["proportion_damaged"]

        elif asset_type == "airport_terminal":
             
                asset_event["cost_usd_per_m"] = cost.loc[cost["asset_type"] == "airport", "mean_cost_usd_per_sqm"].values[0]*10
                asset_event["area_sqm"] = asset_event.to_crs("EPSG:102022").geometry.area
                damage_curve = sndam.from_csv(curves["airport_terminal_polygons_costs"], intensity_col="flood_depth", damage_col="damage_fraction_mean", comment="#")
                asset_event["proportion_damaged"] = damage_curve.damage_fraction(asset_event["depth_m"])
                asset_event["damage_usd"] = asset_event["area_sqm"] * asset_event["cost_usd_per_sqm"] * asset_event["proportion_damaged"]
                
        elif asset_type == "airport_field":
                
                asset_event["cost_usd_per_m"] = cost.loc[cost["asset_type"] == "airport_field", "mean_cost_usd_per_sqm"].values[0]*10
                asset_event["area_sqm"] = asset_event.to_crs("EPSG:102022").geometry.area
                damage_curve = sndam.from_csv(curves["airport_field_polygons_costs"], intensity_col="flood_depth", damage_col="damage_fraction_mean", comment="#")
                asset_event["proportion_damaged"] = damage_curve.damage_fraction(asset_event["depth_m"])
                asset_event["damage_usd"] = asset_event["area_sqm"] * asset_event["cost_usd_per_sqm"] * asset_event["proportion_damaged"]

        elif asset_type == "maritime":
        

            
        
        # Write one output file per event into the same event=* folder
        out_dir  = os.path.dirname(parquet_path)
        out_path = os.path.join(out_dir, "asset_with_depth.parquet")
        asset_event.to_parquet(out_path, index=False)
    

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s",
        level=logging.INFO
    )

    main()