import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from jba_events.disconnection import (
    Network,
    build_graph,
    centre_source_nodes,
    route_to_nearest_centre,
    served_population,
)
from jba_events.risk import exceedance, exceedance_ranks, expected_annual


def test_exceedance_ranks():
    ranks = exceedance_ranks([0.01, 0.001], 10_000)
    assert ranks == {0.01: 100, 0.001: 10}
    assert exceedance_ranks([0.0001], 100) == {0.0001: 1}  # rarer than the set resolves: largest event
    with pytest.raises(ValueError):
        exceedance_ranks([1.0], 100)


def test_expected_annual_and_exceedance_by_key():
    con = duckdb.connect()
    events = pd.DataFrame(
        {"region": ["a"] * 5 + ["b"], "event_id": range(6), "loss": [5.0, 1.0, 4.0, 2.0, 3.0, 10.0]}
    )
    con.register("events", events)
    ead = expected_annual(con, "events", ["region"], ["loss"], 10).set_index("region")
    assert ead.loc["a", "expected_annual_loss"] == pytest.approx(1.5)
    # 10 years: p = 1 - exp(-0.3) is the third largest; region b has only one event
    p = 1 - np.exp(-0.3)
    exc = exceedance(con, "events", ["region"], ["loss"], 10, [p]).set_index("region")
    assert exc.loc["a", "loss"] == 3.0
    assert exc.loc["b", "loss"] == 0.0


@pytest.fixture
def network():
    """A chain w - m - c1 with a spur m - c2; centres at c1 and c2.

        w --e1-- m --e2-- c1
                 |
                 e3
                 |
                 c2
    """
    edges = gpd.GeoDataFrame(
        {"id": ["e1", "e2", "e3"]},
        geometry=[
            LineString([(0, 0), (0.1, 0)]),
            LineString([(0.1, 0), (0.2, 0)]),
            LineString([(0.1, 0), (0.1, -0.3)]),
        ],
        crs="EPSG:4326",
    )
    edge_table, nodes = build_graph(edges)
    centres = gpd.GeoDataFrame(
        {"uc": [1, 2]}, geometry=[box(0.19, -0.01, 0.21, 0.01), box(0.09, -0.31, 0.11, -0.29)], crs="EPSG:4326"
    )
    sources = centre_source_nodes(nodes, centres, "uc", 5000)
    routes = route_to_nearest_centre(edge_table, len(nodes), sources)
    cells = pd.DataFrame(
        {"lon": [0.0, 0.1], "lat": [0.001, 0.001], "population": [10.0, 20.0], "adm1": ["r", "r"], "adm0": ["c", "c"]}
    )
    cells, population = served_population(cells, nodes, routes, 5000)
    return Network(edge_table, routes, sources, population), nodes, routes


def test_baseline_routes_to_nearest_centre(network):
    net, nodes, routes = network
    assert len(nodes) == 4
    w = nodes[(nodes.lon == 0) & (nodes.lat == 0)].node.iloc[0]
    assert routes.centre_id[w] == 1  # c1 is 0.2 degrees away, c2 is 0.4
    assert routes.distance_m[w] == pytest.approx(0.2 * 111_320, rel=0.01)
    assert net.population.population.sum() == 30


def test_disconnected_versus_isolated(network):
    net, _, _ = network
    assert net.event_table([]).empty
    # e2 fails: everyone loses their nearest centre (c1) but can still reach c2
    table = net.event_table(["e2"])
    assert table.population_disconnected.sum() == 30 and table.population_isolated.sum() == 0
    # e1 fails: only the western 10 are cut off, from every centre
    table = net.event_table(["e1"])
    assert table.population_disconnected.sum() == 10 and table.population_isolated.sum() == 10
    # e2 and e3 fail: the 20 at m are isolated too
    table = net.event_table(["e2", "e3"])
    assert table.population_isolated.sum() == 30
