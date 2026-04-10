import sys
import os
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
import glob

from shapely.set_operations import difference
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
    terminals   = gpd.read_file(os.path.join(incoming, "infrastructure/africa_airports_terminal_polygons.gpkg"))
    osm_all = gpd.read_parquet(os.path.join(incoming, "infrastructure/africa_airports_osm.parquet"))
    CRS = "ESRI:102022"  
    terminals  = terminals.to_crs(CRS)
    osm_all = osm_all.to_crs(CRS)
    terminals_union = terminals.geometry.union_all()
    
    joined = gpd.sjoin(
        osm_all,           # left  — features you want to enrich
        terminals,             # right — buffer layer carrying point attributes
        how="inner",        # "inner" keeps only polygons that intersect a buffer
                            # use "left" to keep ALL polygons (NaN if no match)
        predicate="intersects"
    )
    joined = joined[~joined.index.duplicated(keep='first')]
    difference = joined.copy()
    difference['geometry'] = joined.geometry.difference(terminals_union)
    difference["tags.aeroway_osm"] = "field"

    difference.drop(columns=["index_right","tags","feature_id_osm"], inplace=True)
    difference.rename(columns={"feature_id":"feature_id_osm"}, inplace=True)
    terminals = terminals.to_crs("EPSG:4326")
    difference = difference.to_crs("EPSG:4326")


    terminals.to_parquet(os.path.join(incoming, "infrastructure/africa_airport_terminal_polygons_network.geoparquet"))
    difference.to_parquet(os.path.join(incoming, "infrastructure/africa_airport_field_polygons_network.geoparquet"))
    

    
    
   

    

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)
