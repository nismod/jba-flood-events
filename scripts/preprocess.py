import sys
import os
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping, LineString
import numpy as np
import glob
import fsspec

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(script_dir,'..', 'config.json')

    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
    return config

def main(config):
    # Merge all defended areas shp in one .gpkg file
    incoming = config['paths']['incoming_data']
    incoming_defended = os.path.join(incoming, "global_2021", "GlobalDefendedAreas")
    incoming_eventset = os.path.join(incoming, "global_2021","GlobalEventSet")

    output = config['paths']['data']
    output_defended = os.path.join(output, "defended_areas")
    output_events = os.path.join(output, "events")
   
    target_crs = "EPSG:4326" # WGS 84

    defended_files = glob.glob(
        os.path.join(incoming_defended, "**", "*.shp"),
        recursive=True
    )
    if not defended_files:
            return pd.Series([np.nan, np.nan, np.nan, np.nan])
    dfs_defended = [gpd.read_file(f).to_crs(target_crs) for f in defended_files]
    data_defended = pd.concat(dfs_defended, ignore_index=True)
    data_defended.to_file(os.path.join(output_defended, "defended_areas.gpkg"), driver="GPKG")
    
    # Merge all event files with rp by OP by event in one .csv file and the OP info files in one .gpkg file with geometry
    sim_event_files_rp = glob.glob(
        os.path.join(incoming_eventset, "**", "SimEventRp_*.csv"),
        recursive=True
    )
    if not sim_event_files_rp:
            return pd.Series([np.nan, np.nan, np.nan, np.nan])
    dfs_sim = [pd.read_csv(f) for f in sim_event_files_rp]
    data_sim = pd.concat(dfs_sim, ignore_index=True)
    data_sim.to_csv(os.path.join(output_events, "SimEventRp.csv"))

    obs_event_files_rp = glob.glob(
        os.path.join(incoming_eventset, "**", "ObsEventRp_*.csv"),
        recursive=True
    )
    if not obs_event_files_rp:
            return pd.Series([np.nan, np.nan, np.nan, np.nan])
    dfs_obs = [pd.read_csv(f) for f in obs_event_files_rp]
    data_obs = pd.concat(dfs_obs, ignore_index=True)
    data_obs.to_csv(os.path.join(output_events, "ObsEventRp.csv"))
    

    event_files_info = glob.glob(
        os.path.join(incoming_eventset, "**", "RiverOpInfo.csv"),
        recursive=True
    )
    if not event_files_info:
            return pd.Series([np.nan, np.nan, np.nan, np.nan])
    dfs_info = [pd.read_csv(f) for f in event_files_info]
    data_info = pd.concat(dfs_info, ignore_index=True)
    data_info_with_geom = gpd.GeoDataFrame(
        data_info, 
        geometry=gpd.points_from_xy(data_info["op.lon"], data_info["op.lat"]),
        crs= "EPSG:4326" # Sets coordinate reference system to WGS 84 
    )
    data_info_with_geom.to_file(os.path.join(output_events, "RiverOpInfo.gpkg"), driver="GPKG")
   

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
