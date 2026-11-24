"""Split asset geometries on a window of the hazard grid.

Each output piece lies within a single grid cell and carries its local cell
index (`i`, `j`) on the window. Callers convert to global indices by adding the
window offsets.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from snail.core.intersections import split_linestring

from floodlib.grid import GridWindow

GEOMETRY_KINDS = {
    "edges": "line",
    "nodes": "point",
    "polygons": "polygon",
}

ALLOWED_TYPES = {
    "line": {"LineString", "MultiLineString"},
    "point": {"Point", "MultiPoint"},
    "polygon": {"Polygon", "MultiPolygon"},
}


def layer_kind(layer: str) -> str:
    """Geometry kind from a layer name such as `road_edges` or `airport_field_polygons`."""
    suffix = layer.rsplit("_", 1)[-1]
    try:
        return GEOMETRY_KINDS[suffix]
    except KeyError:
        raise ValueError(
            f"Layer '{layer}' must end in one of {sorted(GEOMETRY_KINDS)}"
        ) from None


def layer_asset_type(layer: str) -> str:
    """Asset type, used to find damage curves, e.g. `road_edges` -> `road`."""
    return layer.rsplit("_", 1)[0]


def check_geometry_types(gdf: gpd.GeoDataFrame, kind: str, name: str = ""):
    types = set(gdf.geometry.geom_type.dropna().unique())
    unexpected = types - ALLOWED_TYPES[kind]
    if unexpected:
        raise ValueError(
            f"{name}: expected only {sorted(ALLOWED_TYPES[kind])}, found {sorted(unexpected)}"
        )


def split_on_window(features: gpd.GeoDataFrame, window: GridWindow, kind: str) -> gpd.GeoDataFrame:
    """Split single-part `features` on `window`.

    Returns the input columns plus `i`, `j` (local cell index). Pieces outside
    the window are dropped.
    """
    if kind == "point":
        pieces = features.copy()
    elif kind == "line":
        pieces = _split_lines(features, window)
    elif kind == "polygon":
        pieces = _split_polygons(features, window)
    else:
        raise ValueError(kind)

    if pieces.empty:
        out = pieces.copy()
        out["i"] = np.array([], dtype=np.int64)
        out["j"] = np.array([], dtype=np.int64)
        return out

    if kind == "polygon":
        # cell already known from the box each piece was cut with
        return pieces[(pieces.i >= 0) & (pieces.j >= 0)].reset_index(drop=True)

    if kind == "line":
        ref = shapely.line_interpolate_point(pieces.geometry.values, 0.5, normalized=True)
    else:
        ref = pieces.geometry.values
    i, j = window.cell_indices(shapely.get_x(ref), shapely.get_y(ref))
    pieces["i"] = i
    pieces["j"] = j
    return pieces[(pieces.i >= 0) & (pieces.j >= 0)].reset_index(drop=True)


def _split_lines(features: gpd.GeoDataFrame, window: GridWindow) -> gpd.GeoDataFrame:
    # Cut to the window rectangle first, so the grid split only sees the
    # parts of each line that can fall in a window cell.
    clipped = shapely.clip_by_rect(np.asarray(features.geometry.values), *window.bounds)
    lines = features.set_geometry(gpd.GeoSeries(clipped, index=features.index, crs=features.crs))
    lines = lines[~(lines.geometry.is_empty | lines.geometry.isna())]
    lines = lines.explode(ignore_index=True)
    lines = lines[lines.geometry.geom_type == "LineString"].reset_index(drop=True)

    transform = list(tuple(window.affine)[:6])
    source_idx = []
    geoms = []
    for n, geom in enumerate(lines.geometry.values):
        for piece in split_linestring(geom, window.width, window.height, transform):
            # snail can return zero-length pieces on cell edges
            if piece.length > 0:
                source_idx.append(n)
                geoms.append(piece)

    logging.info("Split %d lines into %d pieces", len(lines), len(geoms))
    attrs = lines.drop(columns=lines.geometry.name).iloc[source_idx].reset_index(drop=True)
    return gpd.GeoDataFrame(attrs, geometry=geoms, crs=features.crs)


def _split_polygons(features: gpd.GeoDataFrame, window: GridWindow) -> gpd.GeoDataFrame:
    """Intersect each polygon with only the cells covered by its bounding box."""
    a, _, c, _, e, f = tuple(window.affine)[:6]
    geoms = features.geometry.values
    shapely.prepare(geoms)

    source_idx, cell_i, cell_j, pieces = [], [], [], []
    for n, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        i0 = max(0, int(np.floor((minx - c) / a)))
        i1 = min(window.width - 1, int(np.ceil((maxx - c) / a)) - 1)
        j0 = max(0, int(np.floor((maxy - f) / e)))
        j1 = min(window.height - 1, int(np.ceil((miny - f) / e)) - 1)
        if i1 < i0 or j1 < j0:
            continue
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
        ii, jj = ii.ravel(), jj.ravel()
        boxes = shapely.box(c + ii * a, f + (jj + 1) * e, c + (ii + 1) * a, f + jj * e)
        hit = shapely.intersects(geom, boxes)
        if not hit.any():
            continue
        cut = shapely.intersection(geom, boxes[hit])
        keep = shapely.area(cut) > 0
        source_idx.extend([n] * int(keep.sum()))
        cell_i.append(ii[hit][keep])
        cell_j.append(jj[hit][keep])
        pieces.extend(cut[keep])

    logging.info("Split %d polygons into %d pieces", len(features), len(pieces))
    attrs = features.drop(columns=features.geometry.name).iloc[source_idx].reset_index(drop=True)
    out = gpd.GeoDataFrame(attrs, geometry=list(pieces), crs=features.crs)
    out["i"] = np.concatenate(cell_i) if cell_i else np.array([], dtype=np.int64)
    out["j"] = np.concatenate(cell_j) if cell_j else np.array([], dtype=np.int64)
    return out


def piece_sizes(pieces: gpd.GeoDataFrame, kind: str) -> pd.Series:
    """Length (m), area (m2) or count (1) per piece, in a local UTM CRS."""
    if kind == "point":
        return pd.Series(1.0, index=pieces.index)
    if pieces.empty:
        return pd.Series([], index=pieces.index, dtype=float)
    utm = pieces.estimate_utm_crs()
    projected = pieces.geometry.to_crs(utm)
    return projected.length if kind == "line" else projected.area


def representative_points(pieces: gpd.GeoDataFrame, kind: str) -> gpd.GeoSeries:
    """A point on each piece, used to look up defended areas."""
    geoms = pieces.geometry.values
    if kind == "line":
        pts = shapely.line_interpolate_point(geoms, 0.5, normalized=True)
    elif kind == "polygon":
        pts = shapely.point_on_surface(geoms)
    else:
        pts = geoms
    return gpd.GeoSeries(pts, index=pieces.index, crs=pieces.crs)
