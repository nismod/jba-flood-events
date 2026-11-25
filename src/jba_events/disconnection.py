"""Population disconnected and detour distance for flood events (draft).

Baseline (once):

1. Build a network from a road layer with snkit, joining line ends within
   `node_tolerance_m`.
2. Add a node for each populated grid cell and each urban centre, and blend
   them into the network: each is projected onto its nearest road within
   `max_snap_m`, the road is split there, and a connector edge is added.
   Split pieces keep the asset id of their road; connectors have none and
   never fail.
3. Urban centres are the sources: their blended nodes, and every network
   node inside a centre polygon.
4. One multi-source shortest-path pass gives each node its distance to the
   nearest centre, d0, and a shortest-path tree.

Per event (or per asset, for static criticality):

5. Remove every edge of each failed asset.
6. Only nodes whose tree path to a centre used a removed edge can change:
   the subtrees below removed tree edges. Distances are recomputed for those
   nodes alone (see `RouteNetwork.reroute`), which gives exactly the same
   result as re-running the whole shortest-path search.
7. A served population node is *disconnected* if it can no longer reach any
   centre, and *detoured* if it can, but by a longer route. The detour is
   d1 - d0, counted in person-km.

Limits of this draft: distance is along the network, with no speeds or
congestion; centres are treated as equivalent destinations; only one network
layer is used.
"""

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.windows
import scipy.sparse as sp
import shapely
from pyproj import Geod, Transformer
from scipy.sparse.csgraph import depth_first_order, dijkstra

# Distances shorter than this (m) are rounding, not detours
DETOUR_EPSILON_M = 1.0
# Lower bound on edge weights, so zero-length edges still count as edges
MIN_EDGE_M = 1e-3

KIND_ROAD, KIND_POPULATION, KIND_CENTRE = "road", "population", "centre"


# ---------------------------------------------------------------------------
# Inputs


def read_population_cells(path, bounds_lonlat, admin=None, adm1_column=None, adm0_column=None) -> pd.DataFrame:
    """Populated cells of a population raster within lon/lat bounds.

    Returns lon, lat (cell centre), population, and adm1, adm0 of the admin
    polygon containing the cell centre (null if none).
    """
    with rasterio.open(path) as src:
        to_raster = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        bounds = to_raster.transform_bounds(*bounds_lonlat, densify_pts=101)
        window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths().intersection(
            rasterio.windows.Window(0, 0, src.width, src.height)
        )
        data = src.read(1, window=window).astype(np.float64)
        transform = src.window_transform(window)
        nodata, crs = src.nodata, src.crs
    if nodata is not None:
        data[data == nodata] = 0
    data[~np.isfinite(data) | (data < 0)] = 0
    rows, cols = np.nonzero(data > 0)
    x, y = rasterio.transform.xy(transform, rows, cols)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(np.asarray(x), np.asarray(y))
    cells = pd.DataFrame({"lon": lon, "lat": lat, "population": data[rows, cols]})

    # admin by cell centre, rasterised onto the population grid
    cells["adm1"] = None
    cells["adm0"] = None
    if admin is not None and len(admin) and len(cells):
        admin = admin.to_crs(crs).reset_index(drop=True)
        index = rasterio.features.rasterize(
            ((geom, i + 1) for i, geom in enumerate(admin.geometry)),
            out_shape=data.shape,
            transform=transform,
            fill=0,
            dtype="int32",
        )[rows, cols]
        inside = index > 0
        cells.loc[inside, "adm1"] = admin[adm1_column].astype(str).to_numpy()[index[inside] - 1]
        cells.loc[inside, "adm0"] = admin[adm0_column].astype(str).to_numpy()[index[inside] - 1]
    logging.info("%d populated cells, population %.0f", len(cells), cells.population.sum())
    return cells


# ---------------------------------------------------------------------------
# Building the network with snkit


def blend_network(roads, cells, centres, centre_id_column, network_crs, max_snap_m, node_tolerance_m=1.0):
    """Road network with population cells and urban centres blended into it.

    roads: single-part lines with an `id` column (the asset id).
    cells: lon, lat, population, adm1, adm0.
    centres: urban centre polygons (or points) with `centre_id_column`.

    Returns (nodes, edges) as DataFrames indexed 0..n-1:
      nodes: node, lon, lat, kind, population, adm1, adm0, centre_id
      edges: u, v, asset_id, length_m, plus a `geometry` column of lon/lat lines
    """
    import snkit
    from snkit.arrays import lines_of, points_of

    roads = roads[["id", "geometry"]].rename(columns={"id": "asset_id"}).to_crs(network_crs)
    roads["asset_id"] = roads["asset_id"].astype(str)
    network = snkit.from_edges(roads, tolerance=node_tolerance_m)
    logging.info("snkit network from roads: %d nodes, %d edges", len(network.nodes), len(network.edges))

    to_network = Transformer.from_crs("EPSG:4326", network_crs, always_xy=True)
    px, py = to_network.transform(cells.lon.to_numpy(), cells.lat.to_numpy())
    centres = centres[[centre_id_column, "geometry"]].to_crs(network_crs).reset_index(drop=True)
    centre_points = centres.geometry.representative_point()
    points = pd.DataFrame(
        {
            "kind": [KIND_POPULATION] * len(cells) + [KIND_CENTRE] * len(centres),
            "population": np.r_[cells.population.to_numpy(dtype=float), np.zeros(len(centres))],
            "adm1": np.r_[cells.adm1.to_numpy(dtype=object), np.full(len(centres), None, dtype=object)],
            "adm0": np.r_[cells.adm0.to_numpy(dtype=object), np.full(len(centres), None, dtype=object)],
            "centre_id": np.r_[np.full(len(cells), None, dtype=object), centres[centre_id_column].astype(str).to_numpy()],
        }
    )
    first_id = int(network.nodes["id"].max()) + 1 if len(network.nodes) else 0
    points["id"] = np.arange(first_id, first_id + len(points), dtype=np.uint64)
    points = gpd.GeoDataFrame(
        points,
        geometry=shapely.points(np.r_[px, centre_points.x.to_numpy()], np.r_[py, centre_points.y.to_numpy()]),
        crs=network_crs,
    )
    road_nodes = network.nodes[["id", "geometry"]].assign(kind=KIND_ROAD)
    road_nodes["id"] = road_nodes["id"].astype(np.uint64)
    nodes = gpd.GeoDataFrame(pd.concat([road_nodes, points], ignore_index=True), geometry="geometry", crs=network_crs)
    network = snkit.from_topology(nodes, network.edges[["id", "from_id", "to_id", "asset_id", "geometry"]])

    # blend: project each point onto its nearest road, split there, add a connector
    is_point = network.nodes["kind"].isin([KIND_POPULATION, KIND_CENTRE]).to_numpy()
    blended = snkit.link_nodes_to_edges(
        snkit.to_arrays(network), how="nearest", max_distance=max_snap_m, nodes=is_point, tolerance=node_tolerance_m
    )
    logging.info("Blended %d points: %d nodes, %d edges", is_point.sum(), len(blended.nodes), len(blended.edges))

    # dense node index; coordinates back to lon/lat
    node_ids = blended.nodes["id"].to_numpy(dtype=np.uint64)
    order = np.argsort(node_ids)
    to_lonlat = Transformer.from_crs(network_crs, "EPSG:4326", always_xy=True)
    xy = points_of(blended.nodes).xy
    lon, lat = to_lonlat.transform(xy[:, 0], xy[:, 1])
    node_table = pd.DataFrame(
        {
            "node": np.arange(len(node_ids)),
            "lon": lon,
            "lat": lat,
            "kind": blended.nodes["kind"].fillna(KIND_ROAD).to_numpy(dtype=object),
            "population": blended.nodes["population"].fillna(0).to_numpy(dtype=float),
            "adm1": blended.nodes["adm1"].to_numpy(dtype=object),
            "adm0": blended.nodes["adm0"].to_numpy(dtype=object),
            "centre_id": blended.nodes["centre_id"].to_numpy(dtype=object),
        }
    )

    def dense(ids):
        return order[np.searchsorted(node_ids, np.asarray(ids, dtype=np.uint64), sorter=order)]

    lines = lines_of(blended.edges)
    coords = np.column_stack(to_lonlat.transform(lines.coords[:, 0], lines.coords[:, 1]))
    edge_table = pd.DataFrame(
        {
            "u": dense(blended.edges["from_id"]),
            "v": dense(blended.edges["to_id"]),
            "asset_id": blended.edges["asset_id"].to_numpy(dtype=object),
            "length_m": geodesic_lengths(coords, lines.offsets),
        }
    )
    edge_table["geometry"] = shapely.from_ragged_array(shapely.GeometryType.LINESTRING, coords, (lines.offsets,))
    return node_table, edge_table


def geodesic_lengths(coords: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Length (m) of each lon/lat line held as coordinates and offsets."""
    n_lines = len(offsets) - 1
    if n_lines == 0:
        return np.zeros(0)
    # segments run from each vertex to the next, except across line boundaries
    seg_start = np.arange(len(coords) - 1)
    last_vertex = np.zeros(len(coords), dtype=bool)
    last_vertex[offsets[1:] - 1] = True
    seg_start = seg_start[~last_vertex[:-1]]
    _, _, seg_len = Geod(ellps="WGS84").inv(
        coords[seg_start, 0], coords[seg_start, 1], coords[seg_start + 1, 0], coords[seg_start + 1, 1]
    )
    line_of_segment = np.searchsorted(offsets, seg_start, side="right") - 1
    return np.bincount(line_of_segment, weights=seg_len, minlength=n_lines)


def centre_sources(nodes: pd.DataFrame, centres: gpd.GeoDataFrame, centre_id_column: str) -> pd.DataFrame:
    """Table (node, centre_id) of the nodes standing for each urban centre.

    Blended centre nodes, and any node inside a centre polygon.
    """
    blended = nodes[(nodes.kind == KIND_CENTRE)][["node", "centre_id"]]
    centres = centres[[centre_id_column, "geometry"]].to_crs("EPSG:4326")
    polygons = centres[centres.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    lon, lat = nodes.lon.to_numpy(), nodes.lat.to_numpy()
    inside = []
    for centre_id, geom in zip(polygons[centre_id_column].astype(str), polygons.geometry):
        minx, miny, maxx, maxy = geom.bounds
        candidates = np.flatnonzero((lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy))
        hit = candidates[shapely.contains_xy(geom, lon[candidates], lat[candidates])]
        inside.append(pd.DataFrame({"node": hit, "centre_id": centre_id}))
    sources = pd.concat([blended, *inside], ignore_index=True).drop_duplicates("node")
    return sources.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Graph helpers


def min_csr(u, v, w, n) -> sp.csr_matrix:
    """Sparse matrix keeping the smallest weight of parallel (u, v) entries.

    (scipy would otherwise add parallel entries together.)
    """
    u, v, w = np.asarray(u), np.asarray(v), np.asarray(w, dtype=float)
    if len(u) == 0:
        return sp.csr_matrix((n, n))
    order = np.lexsort((w, v, u))
    u, v, w = u[order], v[order], w[order]
    first = np.r_[True, (u[1:] != u[:-1]) | (v[1:] != v[:-1])]
    return sp.csr_matrix((w[first], (u[first], v[first])), shape=(n, n))


def shortest_path_tree(edges: pd.DataFrame, n_nodes: int, sources: np.ndarray):
    """Multi-source shortest paths: (distance, parent, nearest source)."""
    u, v = edges.u.to_numpy(), edges.v.to_numpy()
    w = np.maximum(edges.length_m.to_numpy(dtype=float), MIN_EDGE_M)
    keep = u != v
    graph = min_csr(np.r_[u[keep], v[keep]], np.r_[v[keep], u[keep]], np.r_[w[keep], w[keep]], n_nodes)
    distance, parent, source = dijkstra(
        graph, directed=True, indices=sources, min_only=True, return_predecessors=True
    )
    parent = np.where(parent < 0, -1, parent)
    return distance, parent, source


def preorder_intervals(parent: np.ndarray, reachable: np.ndarray):
    """Preorder position and subtree size of each node in a forest.

    `parent` is -1 at roots and unreachable nodes. Returns (order, tin, size):
    the subtree of node x is `order[tin[x] : tin[x] + size[x]]`. tin is -1
    for unreachable nodes.
    """
    n = len(parent)
    children = np.flatnonzero(parent >= 0)
    roots = np.flatnonzero(reachable & (parent < 0))
    # a virtual root n above every tree root
    tree = sp.csr_matrix(
        (np.ones(len(children) + len(roots)), (np.r_[parent[children], np.full(len(roots), n)], np.r_[children, roots])),
        shape=(n + 1, n + 1),
    )
    order = depth_first_order(tree, n, directed=True, return_predecessors=False)[1:]
    tin = np.full(n, -1, dtype=np.int64)
    tin[order] = np.arange(len(order))

    # depth by pointer doubling, then subtree sizes level by level, deepest first
    ancestor, depth = parent.copy(), (parent >= 0).astype(np.int64)
    while (ancestor >= 0).any():
        step = ancestor >= 0
        depth[step] += depth[ancestor[step]]
        ancestor[step] = ancestor[ancestor[step]]
    size = np.where(reachable, 1, 0).astype(np.int64)
    by_depth = children[np.argsort(-depth[children], kind="stable")]
    levels = np.flatnonzero(np.r_[True, np.diff(depth[by_depth]) != 0, True])
    for start, end in zip(levels[:-1], levels[1:]):
        level = by_depth[start:end]
        np.add.at(size, parent[level], size[level])
    return order, tin, size


# ---------------------------------------------------------------------------
# Re-routing after failures


class RouteNetwork:
    """Network, baseline routes and population, ready for failure analysis."""

    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame):
        n = len(nodes)
        self.n_nodes = n
        self.d0 = nodes.distance_m.to_numpy(dtype=float)
        self.parent = nodes.parent.to_numpy(dtype=np.int64)
        self.tin = nodes.tin.to_numpy(dtype=np.int64)
        in_tree = self.tin >= 0
        self.order = np.empty(in_tree.sum(), dtype=np.int64)
        self.order[self.tin[in_tree]] = np.flatnonzero(in_tree)
        self.size = nodes.subtree_size.to_numpy(dtype=np.int64)
        self.population = np.where(np.isfinite(self.d0), nodes.population.to_numpy(dtype=float), 0.0)
        self.pop_prefix = np.r_[0.0, np.cumsum(self.population[self.order])]
        self.adm1 = nodes.adm1.to_numpy(dtype=object)
        self.adm0 = nodes.adm0.to_numpy(dtype=object)

        u, v = edges.u.to_numpy(), edges.v.to_numpy()
        self.u, self.v = u, v
        self.w = np.maximum(edges.length_m.to_numpy(dtype=float), MIN_EDGE_M)
        # node -> incident (neighbour, weight, edge) in CSR form, both directions
        ends = np.r_[u, v]
        others = np.r_[v, u]
        edge_index = np.r_[np.arange(len(u)), np.arange(len(u))]
        order = np.argsort(ends, kind="stable")
        self.adj_ptr = np.r_[0, np.cumsum(np.bincount(ends, minlength=n))]
        self.adj_node = others[order]
        self.adj_w = np.r_[self.w, self.w][order]
        self.adj_edge = edge_index[order]

        # asset id -> edges (connectors have no asset and never fail)
        assets = edges.asset_id
        has_asset = assets.notna().to_numpy()
        codes, self.asset_ids = pd.factorize(assets[has_asset].astype(str))
        edge_of = np.flatnonzero(has_asset)
        by_asset = np.argsort(codes, kind="stable")
        self.asset_edges = edge_of[by_asset]
        self.asset_ptr = np.r_[0, np.cumsum(np.bincount(codes, minlength=len(self.asset_ids)))]
        self.asset_index = pd.Index(self.asset_ids)
        self._failed = np.zeros(len(u), dtype=bool)

    @classmethod
    def from_baseline(cls, baseline_dir) -> "RouteNetwork":
        """Read the tables written by `workflow/scripts/disconnection_baseline.py`."""
        nodes = pd.read_parquet(os.path.join(baseline_dir, "nodes.parquet"))
        edges = pd.read_parquet(
            os.path.join(baseline_dir, "edges.parquet"), columns=["u", "v", "asset_id", "length_m"]
        )
        return cls(nodes, edges)

    def asset_edge_slice(self, i: int) -> np.ndarray:
        """Edges of the i-th asset in `asset_ids`."""
        return self.asset_edges[self.asset_ptr[i] : self.asset_ptr[i + 1]]

    def edges_of_assets(self, asset_ids) -> np.ndarray:
        idx = self.asset_index.get_indexer(pd.Index(np.asarray(asset_ids, dtype=str)))
        idx = idx[idx >= 0]
        if len(idx) == 0:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([self.asset_edge_slice(i) for i in idx])

    def affected_intervals(self, failed_edges: np.ndarray):
        """Merged preorder intervals of the subtrees cut off by removing edges."""
        u, v = self.u[failed_edges], self.v[failed_edges]
        child = np.r_[v[self.parent[v] == u], u[self.parent[u] == v]]
        child = child[self.tin[child] >= 0]
        if len(child) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        start = self.tin[child]
        end = start + self.size[child]
        order = np.argsort(start)
        start, end = start[order], end[order]
        # tree intervals nest or are disjoint: keep those not inside an earlier one
        outer = end > np.r_[-1, np.maximum.accumulate(end)[:-1]]
        return np.column_stack([start[outer], end[outer]])

    def population_at_risk(self, intervals) -> float:
        """Baseline population whose route to a centre uses a removed edge."""
        return float((self.pop_prefix[intervals[:, 1]] - self.pop_prefix[intervals[:, 0]]).sum())

    def reroute(self, failed_edges: np.ndarray, intervals=None):
        """Nodes whose route used a removed edge, and their new distances (inf if none).

        Exact: nodes outside the cut-off subtrees keep their baseline routes,
        so a cut-off node's best new route runs through cut-off nodes until it
        first reaches a node outside, and continues on that node's baseline
        route. A search from a virtual source joined to each cut-off node at
        `d0(outside neighbour) + edge length` therefore finds every new
        distance.
        """
        if intervals is None:
            intervals = self.affected_intervals(failed_edges)
        if len(intervals) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        nodes = np.concatenate([self.order[s:e] for s, e in intervals])
        k = len(nodes)
        local = np.full(self.n_nodes, -1, dtype=np.int64)
        local[nodes] = np.arange(k)

        self._failed[failed_edges] = True
        counts = self.adj_ptr[nodes + 1] - self.adj_ptr[nodes]
        entry = np.repeat(self.adj_ptr[nodes] - np.r_[0, np.cumsum(counts)[:-1]], counts) + np.arange(counts.sum())
        source = np.repeat(np.arange(k), counts)
        neighbour, weight = self.adj_node[entry], self.adj_w[entry]
        ok = ~self._failed[self.adj_edge[entry]]
        self._failed[failed_edges] = False

        source, neighbour, weight = source[ok], neighbour[ok], weight[ok]
        inside = local[neighbour] >= 0
        outside_d = self.d0[neighbour[~inside]] + weight[~inside]
        finite = np.isfinite(outside_d)
        graph = min_csr(
            np.r_[source[inside], np.full(finite.sum(), k)],
            np.r_[local[neighbour[inside]], source[~inside][finite]],
            np.r_[weight[inside], outside_d[finite]],
            k + 1,
        )
        d1 = dijkstra(graph, directed=True, indices=k)[:k]
        return nodes, d1

    def impact(self, failed_edges: np.ndarray, by_admin: bool = True) -> pd.DataFrame:
        """Population disconnected and detoured after removing edges.

        Rows by (adm1, adm0) if `by_admin`, else one total row; empty if none.
        """
        columns = ["population_disconnected", "population_detoured", "detour_person_km"]
        keys = ["adm1", "adm0"] if by_admin else []
        empty = pd.DataFrame(columns=keys + columns)
        intervals = self.affected_intervals(failed_edges)
        if len(intervals) == 0 or self.population_at_risk(intervals) == 0:
            return empty
        nodes, d1 = self.reroute(failed_edges, intervals)
        pop = self.population[nodes]
        keep = pop > 0
        nodes, d1, pop = nodes[keep], d1[keep], pop[keep]
        extra = d1 - self.d0[nodes]
        disconnected = ~np.isfinite(d1)
        detoured = ~disconnected & (extra > DETOUR_EPSILON_M)
        if not (disconnected.any() or detoured.any()):
            return empty
        rows = pd.DataFrame(
            {
                "population_disconnected": np.where(disconnected, pop, 0.0),
                "population_detoured": np.where(detoured, pop, 0.0),
                "detour_person_km": np.where(detoured, pop * extra / 1000, 0.0),
            }
        )
        rows = rows[disconnected | detoured]
        if not by_admin:
            return rows.sum().to_frame().T
        rows["adm1"] = self.adm1[nodes][disconnected | detoured]
        rows["adm0"] = self.adm0[nodes][disconnected | detoured]
        return rows.groupby(keys, dropna=False)[columns].sum().reset_index()


def build_baseline(nodes: pd.DataFrame, edges: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    """Add baseline routing columns to the node table.

    distance_m, parent, nearest_centre_id, is_source, tin, subtree_size, and
    status for population nodes: served, too_far (not blended: no road within
    the snap distance) or no_route (on a part of the network with no centre).
    """
    n = len(nodes)
    distance, parent, source = shortest_path_tree(edges, n, sources.node.to_numpy())
    reachable = np.isfinite(distance)
    order, tin, size = preorder_intervals(parent, reachable)
    centre_of = sources.set_index("node").centre_id
    nearest = np.full(n, None, dtype=object)
    nearest[reachable] = centre_of.reindex(source[reachable]).to_numpy()

    degree = np.bincount(np.r_[edges.u.to_numpy(), edges.v.to_numpy()], minlength=n)
    is_pop = (nodes.kind == KIND_POPULATION).to_numpy()
    status = np.full(n, None, dtype=object)
    status[is_pop] = np.select(
        [degree[is_pop] == 0, ~reachable[is_pop]], ["too_far", "no_route"], default="served"
    )
    out = nodes.copy()
    out["distance_m"] = distance
    out["parent"] = parent
    out["nearest_centre_id"] = nearest
    out["is_source"] = np.isin(np.arange(n), sources.node.to_numpy())
    out["tin"] = tin
    out["subtree_size"] = size
    out["status"] = status
    return out
