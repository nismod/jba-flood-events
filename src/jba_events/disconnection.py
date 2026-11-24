"""Draft "population disconnected" metric for flood events.

Baseline (once):

1. Build a graph from a network layer (roads). Nodes are the distinct edge
   endpoints, so the network must be noded at edge ends.
2. Urban centres are sources: every network node inside a centre polygon,
   or the node nearest the centre if none is inside.
3. One multi-source shortest-path pass gives each node its nearest centre
   by network distance.
4. Each populated grid cell is snapped to its nearest node, within a snap
   limit. Its population is "served" if that node can reach a centre.

Per event:

5. Remove every edge of an asset whose flood depth reaches the threshold.
6. A served node is *disconnected* if it is no longer in the same connected
   component as its baseline nearest centre. It is *isolated* if its
   component contains no urban centre at all (a subset of disconnected).
7. Sum the population of disconnected and isolated nodes by admin region.

Limits of this draft: it measures loss of any route, not longer journeys or
congestion; a cell is flooded-in only through its snapped node; centres are
treated as equivalent destinations; and only one network layer is used.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
import scipy.sparse as sp
import shapely
from pyproj import Geod, Transformer
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

EARTH_RADIUS_M = 6_371_008.8
COORD_DECIMALS = 7


def _xyz(lon, lat) -> np.ndarray:
    lon, lat = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def nearest_nodes(node_lon, node_lat, lon, lat):
    """Index of, and great-circle distance (m) to, the nearest node for each point."""
    tree = cKDTree(_xyz(node_lon, node_lat))
    chord, index = tree.query(_xyz(lon, lat))
    return index, 2 * EARTH_RADIUS_M * np.arcsin(np.clip(chord / 2, 0, 1))


def build_graph(edges: gpd.GeoDataFrame):
    """Edge and node tables from single-part line features with an `id` column.

    Returns (edge_table[u, v, id, length_m], node_table[node, lon, lat]).
    Parts of one asset share its id, so they fail together.
    """
    edges = edges.to_crs("EPSG:4326")
    geoms = np.asarray(edges.geometry.values)
    start = shapely.get_coordinates(shapely.get_point(geoms, 0))
    end = shapely.get_coordinates(shapely.get_point(geoms, -1))
    coords = np.round(np.vstack([start, end]), COORD_DECIMALS)
    unique, inverse = np.unique(coords, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    n = len(edges)
    geod = Geod(ellps="WGS84")
    length = np.array([geod.geometry_length(g) for g in geoms], dtype=float)
    edge_table = pd.DataFrame(
        {"u": inverse[:n].astype(np.int64), "v": inverse[n:].astype(np.int64), "id": edges["id"].to_numpy(), "length_m": length}
    )
    node_table = pd.DataFrame({"node": np.arange(len(unique)), "lon": unique[:, 0], "lat": unique[:, 1]})
    logging.info("Network graph: %d nodes, %d edges", len(node_table), len(edge_table))
    return edge_table, node_table


def adjacency(u, v, weight, n_nodes) -> sp.csr_matrix:
    return sp.csr_matrix((weight, (u, v)), shape=(n_nodes, n_nodes))


def centre_source_nodes(nodes: pd.DataFrame, centres: gpd.GeoDataFrame, id_column: str, max_snap_m: float) -> pd.DataFrame:
    """Table (node, centre_id) of the network nodes that stand for each urban centre."""
    centres = centres[[id_column, "geometry"]].to_crs("EPSG:4326").reset_index(drop=True)
    node_points = gpd.GeoDataFrame(
        nodes[["node"]], geometry=gpd.points_from_xy(nodes.lon, nodes.lat), crs="EPSG:4326"
    )
    polygons = centres[centres.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    inside = gpd.sjoin(node_points, polygons, predicate="within", how="inner")[["node", id_column]]

    # centres with no node inside (or point centres): the nearest node, if close enough
    rest = centres[~centres[id_column].isin(inside[id_column])]
    points = rest.geometry.representative_point()
    index, distance = nearest_nodes(nodes.lon.values, nodes.lat.values, points.x.values, points.y.values)
    near = pd.DataFrame({"node": nodes.node.values[index], id_column: rest[id_column].values})[distance <= max_snap_m]

    sources = pd.concat([inside, near], ignore_index=True).drop_duplicates("node")
    logging.info(
        "%d of %d urban centres on the network (%d source nodes)",
        sources[id_column].nunique(), len(centres), len(sources),
    )
    return sources.rename(columns={id_column: "centre_id"}).reset_index(drop=True)


def route_to_nearest_centre(edge_table: pd.DataFrame, n_nodes: int, sources: pd.DataFrame) -> pd.DataFrame:
    """Per node: network distance to, and identity of, its nearest urban centre.

    Returns node, distance_m (inf if unreachable), centre_node (-1), centre_id (null).
    """
    graph = adjacency(edge_table.u, edge_table.v, np.maximum(edge_table.length_m, 1e-3), n_nodes)
    distance, _, source = dijkstra(
        graph, directed=False, indices=sources.node.to_numpy(), min_only=True, return_predecessors=True
    )
    centre_of = sources.set_index("node").centre_id
    reachable = source >= 0
    centre_id = np.full(n_nodes, None, dtype=object)
    centre_id[reachable] = centre_of.reindex(source[reachable]).to_numpy()
    return pd.DataFrame(
        {"node": np.arange(n_nodes), "distance_m": distance, "centre_node": np.where(reachable, source, -1), "centre_id": centre_id}
    )


def read_population_cells(path, bounds_lonlat) -> pd.DataFrame:
    """Populated cells (lon, lat, population) of a population raster within lon/lat bounds."""
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
    logging.info("%d populated cells, population %.0f", len(rows), data[rows, cols].sum())
    return pd.DataFrame({"lon": lon, "lat": lat, "population": data[rows, cols]})


def assign_admin(lon, lat, admin: gpd.GeoDataFrame | None, adm1_column, adm0_column) -> pd.DataFrame:
    out = pd.DataFrame({"adm1": [None] * len(lon), "adm0": [None] * len(lon)}, dtype=object)
    if admin is None or len(lon) == 0:
        return out
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    admin = admin[[adm1_column, adm0_column, "geometry"]].to_crs("EPSG:4326")
    joined = gpd.sjoin(points, admin, predicate="within", how="inner")
    joined = joined[~joined.index.duplicated()]
    out.loc[joined.index, "adm1"] = joined[adm1_column].astype(str).values
    out.loc[joined.index, "adm0"] = joined[adm0_column].astype(str).values
    return out


def served_population(cells: pd.DataFrame, nodes: pd.DataFrame, routes: pd.DataFrame, max_snap_m: float):
    """Snap cells to nodes and total the served population per (node, adm1, adm0).

    Returns (cells with node, snap_distance_m and status, population table).
    status is `served`, `too_far` (from the network) or `no_route` (to any centre).
    """
    index, distance = nearest_nodes(nodes.lon.values, nodes.lat.values, cells.lon.values, cells.lat.values)
    cells = cells.copy()
    cells["node"] = nodes.node.values[index]
    cells["snap_distance_m"] = distance
    reachable = np.isfinite(routes.distance_m.to_numpy())[cells.node.to_numpy()]
    cells["status"] = np.select([distance > max_snap_m, ~reachable], ["too_far", "no_route"], default="served")
    served = cells[cells.status == "served"]
    population = (
        served.groupby(["node", "adm1", "adm0"], dropna=False)["population"].sum().reset_index()
    )
    return cells, population


class Network:
    """Graph and served population, ready for per-event connectivity checks."""

    def __init__(self, edge_table: pd.DataFrame, routes: pd.DataFrame, sources: pd.DataFrame, population: pd.DataFrame):
        self.n_nodes = len(routes)
        self.u = edge_table.u.to_numpy()
        self.v = edge_table.v.to_numpy()
        self.edge_asset = edge_table["id"].to_numpy()
        self.source_nodes = sources.node.to_numpy()
        self.population = population.reset_index(drop=True)
        self.pop_node = self.population.node.to_numpy()
        self.pop_centre_node = routes.centre_node.to_numpy()[self.pop_node]

    def disconnected(self, failed_assets) -> tuple[np.ndarray, np.ndarray]:
        """Boolean arrays over the population table rows: (disconnected, isolated)."""
        keep = ~pd.Series(self.edge_asset).isin(failed_assets).to_numpy()
        graph = sp.csr_matrix(
            (np.ones(keep.sum(), dtype=np.int8), (self.u[keep], self.v[keep])), shape=(self.n_nodes, self.n_nodes)
        )
        _, component = connected_components(graph, directed=False)
        has_centre = np.zeros(component.max() + 1, dtype=bool)
        has_centre[component[self.source_nodes]] = True
        disconnected = component[self.pop_node] != component[self.pop_centre_node]
        isolated = ~has_centre[component[self.pop_node]]
        return disconnected, isolated

    def event_table(self, failed_assets) -> pd.DataFrame:
        """Population disconnected and isolated by (adm1, adm0), for regions affected."""
        disconnected, isolated = self.disconnected(failed_assets)
        if not disconnected.any():
            return pd.DataFrame(columns=["adm1", "adm0", "population_disconnected", "population_isolated"])
        rows = self.population[disconnected].assign(
            population_disconnected=lambda d: d.population,
            population_isolated=lambda d: d.population * isolated[disconnected],
        )
        return (
            rows.groupby(["adm1", "adm0"], dropna=False)[["population_disconnected", "population_isolated"]]
            .sum()
            .reset_index()
        )
