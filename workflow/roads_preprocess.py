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



def main(config):
    incoming_data_path = config['paths']['incoming_data']
    processed_data_path = config['paths']['data']
    
    epsg_meters = 3395 # To convert geometries to measure distances in meters
   
    # Read the road edges data for Africa

    road_id_column = "id"
    node_id_column = "id"
    road_type_column = "tag_highway"
    # main_road_types = ["trunk","motorway","primary","secondary"]
    # """
    # Assuming that the starting point is the road network with the main highways
    # Because main corridors should be along main highways
    # """
    
    road_edges = gpd.read_parquet(os.path.join(
                            incoming_data_path,
                            "infrastructure",
                            "edges_with_topology.geoparquet"))
    road_nodes = gpd.read_parquet(os.path.join(
                            incoming_data_path,
                            "infrastructure",
                            "nodes_with_topology.geoparquet"))

    

    main_edges = gpd.read_parquet(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "africa_roads_network.gpkg"),layer="edges")
    main_nodes = gpd.read_parquet(os.path.join(
                                processed_data_path,
                                "infrastructure",
                                "africa_roads_network.gpkg"),layer="nodes")
    
    road_edges = road_edges.to_crs(epsg=epsg_meters)
    road_nodes = road_nodes.to_crs(epsg=epsg_meters) 
    main_edges = main_edges.to_crs(epsg=epsg_meters)
    main_nodes = main_nodes.to_crs(epsg=epsg_meters) 
   
    
    cols_to_use = main_edges.columns.difference(road_edges.columns)

    road_edges = road_edges.merge(
        main_edges[cols_to_use.tolist() + ['osm_way_id']],
        on="osm_way_id",
        how="left"
    )

    print(road_edges) 


    road_nodes.to_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads_nodes_complete.geoparquet"))
    road_edges.to_parquet(os.path.join(
                            processed_data_path,
                            "infrastructure",
                            "africa_roads_edges_complete.geoparquet"))
   
    



if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)