"""Local re-routing must match a full shortest-path search on the damaged network."""

import numpy as np
import pandas as pd
import pytest

from jba_events.disconnection import RouteNetwork, build_baseline, min_csr, preorder_intervals, shortest_path_tree
from scipy.sparse.csgraph import dijkstra


def random_network(seed, n=400, extra=250):
    """A random spanning tree plus extra edges, with parallel edges and several assets per edge set."""
    rng = np.random.default_rng(seed)
    parent = np.r_[-1, [rng.integers(0, i) for i in range(1, n)]]
    u = np.r_[np.arange(1, n), rng.integers(0, n, extra)]
    v = np.r_[parent[1:], rng.integers(0, n, extra)]
    keep = u != v
    u, v = u[keep], v[keep]
    # a few exact parallel duplicates
    dup = rng.choice(len(u), 20, replace=False)
    u, v = np.r_[u, u[dup]], np.r_[v, v[dup]]
    edges = pd.DataFrame({"u": u, "v": v, "length_m": rng.uniform(1, 100, len(u))})
    edges["asset_id"] = (np.arange(len(u)) // 3).astype(str)
    edges.loc[rng.random(len(u)) < 0.1, "asset_id"] = None  # connectors
    nodes = pd.DataFrame(
        {
            "node": np.arange(n),
            "kind": "population",
            "population": rng.integers(0, 5, n).astype(float),
            "adm1": rng.choice(["a", "b"], n),
            "adm0": "c",
        }
    )
    sources = pd.DataFrame({"node": rng.choice(n, 6, replace=False), "centre_id": [str(i) for i in range(6)]})
    return nodes, edges, sources


def full_distances(edges, n, sources, failed):
    keep = np.ones(len(edges), dtype=bool)
    keep[failed] = False
    e = edges[keep]
    u, v, w = e.u.to_numpy(), e.v.to_numpy(), np.maximum(e.length_m.to_numpy(), 1e-3)
    graph = min_csr(np.r_[u, v], np.r_[v, u], np.r_[w, w], n)
    return dijkstra(graph, directed=True, indices=sources, min_only=True)


@pytest.mark.parametrize("seed", range(8))
def test_reroute_matches_full_search(seed):
    nodes, edges, sources = random_network(seed)
    baseline = build_baseline(nodes, edges, sources)
    net = RouteNetwork(baseline, edges)
    rng = np.random.default_rng(seed + 100)
    for _ in range(25):
        assets = rng.choice(net.asset_ids, rng.integers(1, 15), replace=False)
        failed = net.edges_of_assets(assets)
        expected = full_distances(edges, len(nodes), sources.node.to_numpy(), failed)
        changed, d1 = net.reroute(failed)
        actual = baseline.distance_m.to_numpy().copy()
        actual[changed] = d1
        assert np.allclose(np.nan_to_num(actual, posinf=-1), np.nan_to_num(expected, posinf=-1))

        # population impact agrees with a direct count
        table = net.impact(failed, by_admin=False)
        d0 = baseline.distance_m.to_numpy()
        pop = np.where(np.isfinite(d0), nodes.population.to_numpy(), 0)
        lost = pop[np.isfinite(d0) & ~np.isfinite(expected)].sum()
        extra = np.where(np.isfinite(expected) & np.isfinite(d0), expected - d0, 0)
        detoured = extra > 1.0
        if len(table):
            assert table.population_disconnected.iloc[0] == pytest.approx(lost)
            assert table.population_detoured.iloc[0] == pytest.approx(pop[detoured].sum())
            assert table.detour_person_km.iloc[0] == pytest.approx((pop * extra)[detoured].sum() / 1000)
        else:
            assert lost == 0 and pop[detoured].sum() == 0


def test_preorder_intervals_are_subtrees():
    nodes, edges, sources = random_network(3)
    distance, parent, _ = shortest_path_tree(edges, len(nodes), sources.node.to_numpy())
    order, tin, size = preorder_intervals(parent, np.isfinite(distance))
    for x in range(len(nodes)):
        if tin[x] < 0:
            continue
        block = set(order[tin[x] : tin[x] + size[x]])
        # every node in the block has x as an ancestor
        for y in block:
            while y != x:
                y = parent[y]
                assert y >= 0
        assert len(block) == size[x]


def test_min_csr_keeps_smallest_parallel_edge():
    m = min_csr([0, 0, 1], [1, 1, 0], [5.0, 2.0, 3.0], 2)
    assert m[0, 1] == 2.0 and m[1, 0] == 3.0
