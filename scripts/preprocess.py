import sys
import os
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping, LineString
import numpy as np
import glob

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
    output = config['paths']['data']
    output_defended = os.path.join(output, "defended_areas")
    
    print("incoming_defended:", incoming_defended)
    print("exists:", os.path.exists(incoming_defended))
    

    defended_files = glob.glob(
        os.path.join(incoming_defended, "**", "*.shp"),
        recursive=True
    )
    print(defended_files)
    breakpoint()
    if not defended_files:
            return pd.Series([np.nan, np.nan, np.nan, np.nan])
    
    
    
    dfs_defended = [gpd.read_file(f) for f in defended_files]
    data_defended = pd.concat(dfs_defended, ignore_index=True)
    data_defended.to_file(os.path.join(output_defended, "defended_areas.gpkg"), driver="GPKG")

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
