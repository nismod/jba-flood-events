"""Split asset layers on the hazard grid within one HAZ.

A piece belongs to a HAZ if the centre of its grid cell lies inside the HAZ
polygon. HAZs partition space, so every piece is assigned to at most one HAZ.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import CRS, Transformer

from jba_events.grid import GridWindow, HazardGrid
from jba_events.split import layer_kind, piece_sizes, representative_points, split_on_window

SIZE_UNITS = {"line": "m", "polygon": "m2", "point": "count"}


def depth_column(rp: float) -> str:
    return f"depth_rp{rp:g}"


def read_haz(haz_path, haz_id_column, haz_id) -> gpd.GeoDataFrame:
    haz = gpd.read_file(haz_path, columns=[haz_id_column])
    haz = haz[haz[haz_id_column].astype(str) == str(haz_id)]
    if len(haz) == 0:
        raise ValueError(f"HAZ {haz_id} not found in {haz_path}")
    return haz


class HazWindow:
    """One HAZ on the hazard grid: its window, cell mask and mapped depths.

    Built once per HAZ and shared by every asset layer.
    """

    def __init__(self, haz: gpd.GeoDataFrame, grid: HazardGrid, rps, rp_paths):
        self.grid = grid
        self.haz = haz.to_crs(grid.crs)
        self.geometry = self.haz.geometry.union_all()
        self.window: GridWindow = grid.window_for_bounds(self.haz.total_bounds)
        self.in_haz = self.window.mask(self.haz.geometry)
        self.rps = list(rps)
        self.depths = [self.window.read(path) for path in rp_paths]

    def read_bbox(self, path, columns) -> gpd.GeoDataFrame:
        """Read features of a GeoPackage (or similar) intersecting the window."""
        file_crs = CRS.from_user_input(pyogrio.read_info(path)["crs"])
        to_file = Transformer.from_crs(CRS.from_user_input(self.grid.crs), file_crs, always_xy=True)
        bounds = to_file.transform_bounds(*self.window.bounds)
        return gpd.read_file(path, columns=columns, bbox=bounds).to_crs(self.grid.crs)


def split_layer(
    hw: HazWindow,
    layer: str,
    asset_path,
    defended: gpd.GeoDataFrame | None = None,
    sop_column: str = "JBA_SoP",
    admin: gpd.GeoDataFrame | None = None,
    adm1_column: str | None = None,
) -> gpd.GeoDataFrame:
    """Pieces of one prepared asset layer in the HAZ, one row per piece.

    Columns: split_id, id, cost_type, adm1, raster_i, raster_j (global grid
    indices), size, size_unit, sop, depth_rp{RP}..., geometry.
    """
    kind = layer_kind(layer)
    window = hw.window

    assets = gpd.read_parquet(asset_path, columns=["id", "cost_type", "geometry"], bbox=tuple(window.bounds))
    n_bbox = len(assets)
    # The bbox of an irregular HAZ can be much larger than the HAZ itself
    if n_bbox:
        hits = assets.sindex.query(hw.geometry, predicate="intersects")
        assets = assets.iloc[np.sort(hits)].reset_index(drop=True)
    logging.info("%s: %d features in HAZ bbox, %d intersect the HAZ", layer, n_bbox, len(assets))

    pieces = split_on_window(assets, window, kind)
    if len(pieces):
        pieces = pieces[hw.in_haz[pieces.j.values, pieces.i.values]].reset_index(drop=True)

    for rp, data in zip(hw.rps, hw.depths):
        pieces[depth_column(rp)] = data[pieces.j.values, pieces.i.values].astype(np.float32)

    points = representative_points(pieces, kind)
    pieces["raster_i"] = (pieces.i + window.col_off).astype(np.int32)
    pieces["raster_j"] = (pieces.j + window.row_off).astype(np.int32)
    pieces["size"] = piece_sizes(pieces, kind).astype(np.float64)
    pieces["size_unit"] = SIZE_UNITS[kind]
    pieces["sop"] = _piece_sop(points, defended, sop_column)
    pieces["adm1"] = _piece_adm1(points, admin, adm1_column)
    pieces["split_id"] = np.arange(len(pieces), dtype=np.int32)
    pieces["cost_type"] = pieces["cost_type"].astype(object)

    columns = ["split_id", "id", "cost_type", "adm1", "raster_i", "raster_j", "size", "size_unit", "sop"]
    columns += [depth_column(rp) for rp in hw.rps] + ["geometry"]
    logging.info("%s: %d pieces in HAZ cells", layer, len(pieces))
    return gpd.GeoDataFrame(pieces[columns], geometry="geometry", crs=hw.grid.crs)


def _piece_sop(points: gpd.GeoSeries, defended, sop_column) -> pd.Series:
    """Highest standard of protection (as an RP) of the defended areas containing each piece."""
    sop = pd.Series(np.nan, index=points.index)
    if defended is None or defended.empty or points.empty:
        return sop
    defended = defended[defended[sop_column] > 0]
    joined = gpd.sjoin(
        gpd.GeoDataFrame(geometry=points), defended[[sop_column, "geometry"]], predicate="within", how="inner"
    )
    best = joined.groupby(level=0)[sop_column].max()
    sop.loc[best.index] = best.values
    return sop


def _piece_adm1(points: gpd.GeoSeries, admin, adm1_column) -> pd.Series:
    """Admin-1 region containing each piece; null if none."""
    adm1 = pd.Series([None] * len(points), index=points.index, dtype=object)
    if admin is None or admin.empty or points.empty:
        return adm1
    joined = gpd.sjoin(
        gpd.GeoDataFrame(geometry=points), admin[[adm1_column, "geometry"]], predicate="within", how="inner"
    )
    first = joined[~joined.index.duplicated()][adm1_column].astype(str)
    adm1.loc[first.index] = first.values
    return adm1
