import sys
import os
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
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
    # Osm footprints of inland ports merged with AfTS-Db points, then they will be visually checked
    incoming = config['paths']['incoming_data']
    output = config['paths']['data']
    points   = gpd.read_file(os.path.join(incoming, "infrastructure/africa_airport_network_nodes_modified_terminals.gpkg"))
    polygons = gpd.read_file(os.path.join(incoming, "infrastructure/terminals_joined_layer.gpkg"))
    CRS = "ESRI:102022"  
    points   = points.to_crs(CRS)
    polygons = polygons.to_crs(CRS)
    points["geometry"] = points.geometry.buffer(200)
    joined = gpd.sjoin(
        polygons,           # left  — features you want to enrich
        points,             # right — buffer layer carrying point attributes
        how="inner",        # "inner" keeps only polygons that intersect a buffer
                            # use "left" to keep ALL polygons (NaN if no match)
        predicate="intersects"
    )

    joined.to_parquet(os.path.join(incoming, "infrastructure/africa_iww_inland_airports_footprints_joined_modified.geoparquet"))
    print(joined.head())

    
    
   

    

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
