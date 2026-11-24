"""Synthetic data for testing the workflow without the JBA datasets.

Grid: EPSG:4326, 0.01 degree cells, 100 x 60 cells, x 10..11, y 0.4..1.0.
HAZ "A" covers x < 10.5 and "B" covers x >= 10.5, both over the whole grid.
HAZ "C" lies off the grid and far from any OP.

Mapped depths depend only on the column i:
- columns 0-19 are dry at every RP
- columns 20-39 are first wet at RP50
- columns 40-99 are wet from RP20
Depth at a wet RP is rp_index * 0.1 + 0.1 m, so 0.1 at RP20, 0.2 at RP50, ...
"""

import json
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from shapely.geometry import LineString, Point, box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "workflow", "scripts"))

RPS = [20, 50, 100, 200, 500, 1500]
TRANSFORM = Affine(0.01, 0, 10.0, 0, -0.01, 1.0)
WIDTH, HEIGHT = 100, 60


def mapped_depth(i: int, rp_index: int) -> float:
    if i < 20:
        return 0.0
    if i < 40 and rp_index == 0:
        return 0.0
    return rp_index * 0.1 + 0.1


def write_rasters(directory):
    os.makedirs(directory, exist_ok=True)
    for k, rp in enumerate(RPS):
        data = np.array(
            [[mapped_depth(i, k) for i in range(WIDTH)] for _ in range(HEIGHT)], dtype=np.float32
        )
        data[0, :] = -9999  # nodata row
        with rasterio.open(
            os.path.join(directory, f"flrf_ud_Q{rp}.tif"),
            "w",
            driver="GTiff",
            width=WIDTH,
            height=HEIGHT,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=TRANSFORM,
            nodata=-9999,
        ) as dst:
            dst.write(data, 1)


def make_dataset(root):
    data = os.path.join(root, "processed_data")
    write_rasters(os.path.join(data, "hazards"))

    os.makedirs(os.path.join(data, "basins"))
    gpd.GeoDataFrame(
        {"T500_ID": ["A", "B", "C"]},
        geometry=[box(9.0, 0.0, 10.5, 2.0), box(10.5, 0.0, 12.0, 2.0), box(30.0, 0.0, 31.0, 1.0)],
        crs="EPSG:4326",
    ).to_file(os.path.join(data, "basins", "haz_500.gpkg"))

    os.makedirs(os.path.join(data, "events"))
    gpd.GeoDataFrame(
        {"op.id": [1, 2, 3]},
        geometry=[Point(10.2, 0.7), Point(10.7, 0.7), Point(10.8, 0.5)],
        crs="EPSG:4326",
    ).to_file(os.path.join(data, "events", "RiverOpInfo.gpkg"))
    for event_type, rows in {
        "Obs": [(1, 1, 30.0), (1, 2, 100.0), (1, 3, 400.0), (2, 2, 5.0)],
        "Sim": [(10, 1, 2000.0), (11, 1, 15.0), (11, 2, 50.0), (11, 3, 50.0)],
    }.items():
        pd.DataFrame(rows, columns=["event.id", "op.id", "rp"]).to_csv(
            os.path.join(data, "events", f"{event_type}EventRp.csv")
        )

    os.makedirs(os.path.join(data, "infrastructure"))
    # road crosses from HAZ A into HAZ B along y = 0.705 (row 29)
    gpd.GeoDataFrame(
        {"id": ["r1", "r2"], "cost_type": ["roads_paved", None]},
        geometry=[LineString([(10.005, 0.705), (10.995, 0.705)]), LineString([(10.455, 0.9), (10.455, 0.5)])],
        crs="EPSG:4326",
    ).to_parquet(os.path.join(data, "infrastructure", "africa_road_edges_network.parquet"))
    gpd.GeoDataFrame(
        {"id": ["n1", "n2"], "cost_type": [None, None]},
        geometry=[Point(10.455, 0.705), Point(10.105, 0.705)],
        crs="EPSG:4326",
    ).to_parquet(os.path.join(data, "infrastructure", "africa_road_nodes_network.parquet"))
    gpd.GeoDataFrame(
        {"id": ["f1"], "cost_type": ["airport_field"]},
        geometry=[box(10.48, 0.6, 10.53, 0.62)],
        crs="EPSG:4326",
    ).to_parquet(os.path.join(data, "infrastructure", "africa_airport_field_polygons_network.parquet"))

    # defended with SoP 100 over x 10.6..10.7
    os.makedirs(os.path.join(data, "defended_areas"))
    gpd.GeoDataFrame(
        {"JBA_SoP": [100.0, 50.0]},
        geometry=[box(10.6, 0.4, 10.7, 1.0), box(10.65, 0.4, 10.7, 1.0)],
        crs="EPSG:4326",
    ).to_file(os.path.join(data, "defended_areas", "defended_areas.gpkg"))

    os.makedirs(os.path.join(data, "costs"))
    pd.DataFrame({"cost_type": ["roads_paved"], "mean_cost_usd_per_m": [100.0]}).to_csv(
        os.path.join(data, "costs", "road_edges.csv"), index=False
    )
    pd.DataFrame({"cost_type": ["airport_field"], "mean_cost_usd_per_sqm": [10.0]}).to_csv(
        os.path.join(data, "costs", "airport_field_polygons.csv"), index=False
    )
    curve = pd.DataFrame({"flood_depth_m": [0.0, 1.0, 10.0], "damage_fraction_mean": [0.0, 0.5, 1.0]})
    for asset_type, cost_type in [("road", "roads_paved"), ("airport_field", "airport_field")]:
        os.makedirs(os.path.join(data, "curves", asset_type))
        curve.to_csv(os.path.join(data, "curves", asset_type, f"{cost_type}.csv"), index=False)

    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"paths": {"incoming_data": "incoming_data", "data": data, "figures": "figures", "results": "results"}}, fh)
    return data


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("jba")
    return root, make_dataset(str(root))
