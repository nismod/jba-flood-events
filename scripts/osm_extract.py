#!/usr/bin/env python
# coding: utf-8
import sys
import os
import re
import json
import pandas as pd
import igraph as ig
import geopandas as gpd
from utils_new import *
from tqdm import tqdm
tqdm.pandas()
import osmium
from pyrosm import OSM



def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    osm_data = os.path.join(incoming_data_path,"osm", "africa-260219.osm.pbf")
    
    osm = OSM(osm_data)
    airports = osm.get_data_by_custom_criteria(
        custom_filter={"aeroway": ["aerodrome"]},
        keep_nodes=True,
        keep_ways=True
    )
    river_ports = osm.get_data_by_custom_criteria(
        custom_filter={
            "landuse": ["port"],
            "port:type": ["inland_port"]
        },
        keep_nodes=True,
        keep_ways=True,
        keep_relations=True
    )
    sea_ports = osm.get_data_by_custom_criteria(
        custom_filter={
            "landuse": ["port"],
            "port:type": ["seaport"]
        },
        keep_nodes=True,
        keep_ways=True,
        keep_relations=True
    )

    print(airports.head())
    print(river_ports.head())
    print(sea_ports.head())

    river_ports.to_file(os.path.join(processed_data_path,"infrastructure",
        "inland_ports_osm.gpkg"),
        layer="inland_ports",
        driver="GPKG"
    )
    sea_ports.to_file(os.path.join(processed_data_path,"infrastructure",
        "sea_ports_osm.gpkg"),
        layer="sea_ports",
        driver="GPKG"
    )
    airports.to_file(os.path.join(processed_data_path,"infrastructure",
        "airports_osm.gpkg"),
        layer="airports",
        driver="GPKG"
    )

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)