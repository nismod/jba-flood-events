"""Blending population and urban centres into a road network with snkit."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import Geod
from shapely.geometry import LineString, box

from jba_events.disconnection import (
    RouteNetwork,
    blend_network,
    build_baseline,
    centre_sources,
    geodesic_lengths,
)


def test_geodesic_lengths_of_ragged_lines():
    coords = np.array([[0, 0], [1, 0], [1, 1], [5, 5], [5, 6]], dtype=float)
    offsets = np.array([0, 3, 5])
    geod = Geod(ellps="WGS84")
    expected = [
        geod.geometry_length(LineString(coords[0:3])),
        geod.geometry_length(LineString(coords[3:5])),
    ]
    assert geodesic_lengths(coords, offsets) == pytest.approx(expected)


@pytest.fixture
def blended():
    """A T: west-east road w-e, a spur s off its middle, a centre at the east end.

        p1          p2
        w ----m---- e [centre]
              |
              s
              p3
    """
    roads = gpd.GeoDataFrame(
        {"id": ["west", "east", "spur"]},
        geometry=[
            LineString([(0, 0), (0.1, 0)]),
            LineString([(0.1, 0), (0.2, 0)]),
            LineString([(0.1, 0), (0.1, -0.1)]),
        ],
        crs="EPSG:4326",
    )
    cells = pd.DataFrame(
        {
            "lon": [0.02, 0.15, 0.11, 0.5],
            "lat": [0.01, 0.01, -0.09, 0.5],  # the last is far from any road
            "population": [10.0, 20.0, 30.0, 5.0],
            "adm1": ["a", "a", "b", "b"],
            "adm0": "c",
        }
    )
    centres = gpd.GeoDataFrame({"uc": [1]}, geometry=[box(0.19, -0.01, 0.21, 0.01)], crs="EPSG:4326")
    nodes, edges = blend_network(roads, cells, centres, "uc", "ESRI:102023", max_snap_m=5000)
    sources = centre_sources(nodes, centres, "uc")
    baseline = build_baseline(nodes, edges.drop(columns="geometry"), sources)
    return baseline, edges


def test_blend_splits_roads_and_adds_connectors(blended):
    baseline, edges = blended
    # three linked cells and one blended centre: four connectors. Each road is
    # split once by its cell; the centre projects onto the east road's end, so
    # it joins the node there instead of splitting the road again.
    connectors = edges[edges.asset_id.isna()]
    assert len(connectors) == 4
    assert edges.asset_id.value_counts().to_dict() == {"west": 2, "east": 2, "spur": 2}
    assert baseline.status.value_counts().to_dict() == {"served": 3, "too_far": 1}
    # each connector runs about 1.1 km (0.01 degrees) from its cell to its road
    pop_connectors = connectors[connectors.u.isin(baseline[baseline.kind == "population"].node)]
    assert pop_connectors.length_m.to_numpy() == pytest.approx(1106, rel=0.01)


def test_blended_routes_and_failures(blended):
    baseline, edges = blended
    net = RouteNetwork(baseline, edges.drop(columns="geometry"))
    pop = baseline.set_index("population")
    # the cell on the west road routes about 0.18 degrees east to the centre, plus its connector
    assert pop.loc[10.0, "distance_m"] == pytest.approx((0.17 + 0.01) * 111_320 + 1106, rel=0.01)

    # cutting the east road cuts everyone off; the spur only its own cell
    table = net.impact(net.edges_of_assets(["east"]), by_admin=False)
    assert table.population_disconnected.iloc[0] == 60
    table = net.impact(net.edges_of_assets(["spur"])).set_index("adm1")
    assert table.population_disconnected.to_dict() == {"b": 30.0}
    assert net.impact(net.edges_of_assets(["west"]), by_admin=False).population_disconnected.iloc[0] == 10
