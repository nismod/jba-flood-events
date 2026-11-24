"""The continental hazard grid, and windows of it for each HAZ.

All return period (RP) maps must share one grid. Cell indices
(`raster_i` column, `raster_j` row) are always global indices on that grid,
so they are consistent across HAZs and can be joined directly.
"""

import json
import math
from dataclasses import dataclass, asdict

import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
from affine import Affine


@dataclass(frozen=True)
class HazardGrid:
    crs: str
    width: int
    height: int
    transform: tuple  # first six affine coefficients (a, b, c, d, e, f)

    @property
    def affine(self) -> Affine:
        return Affine(*self.transform)

    @classmethod
    def from_raster(cls, path) -> "HazardGrid":
        with rasterio.open(path) as src:
            return cls(
                crs=src.crs.to_string(),
                width=src.width,
                height=src.height,
                transform=tuple(src.transform)[:6],
            )

    @classmethod
    def from_rasters(cls, paths) -> "HazardGrid":
        """Grid shared by all `paths`; raises if any raster differs."""
        paths = list(paths)
        grid = cls.from_raster(paths[0])
        for path in paths[1:]:
            other = cls.from_raster(path)
            if other.crs != grid.crs or (other.width, other.height) != (
                grid.width,
                grid.height,
            ):
                raise ValueError(f"Raster grid mismatch: {path} {other} != {grid}")
            if not np.allclose(other.transform, grid.transform, rtol=0, atol=1e-9):
                raise ValueError(f"Raster transform mismatch: {path} {other} != {grid}")
        if grid.transform[1] != 0 or grid.transform[3] != 0:
            raise ValueError("Rotated or sheared rasters are not supported")
        return grid

    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def from_json(cls, path) -> "HazardGrid":
        with open(path) as fh:
            data = json.load(fh)
        data["transform"] = tuple(data["transform"])
        return cls(**data)

    def window_for_bounds(self, bounds) -> "GridWindow":
        """Smallest window of whole cells covering `bounds`, cut to the grid."""
        minx, miny, maxx, maxy = bounds
        inv = ~self.affine
        cols, rows = zip(*[inv @ (x, y) for x in (minx, maxx) for y in (miny, maxy)])
        col_off = max(0, math.floor(min(cols)))
        row_off = max(0, math.floor(min(rows)))
        col_end = min(self.width, math.ceil(max(cols)))
        row_end = min(self.height, math.ceil(max(rows)))
        return GridWindow(
            grid=self,
            col_off=col_off,
            row_off=row_off,
            width=max(0, col_end - col_off),
            height=max(0, row_end - row_off),
        )


@dataclass(frozen=True)
class GridWindow:
    grid: HazardGrid
    col_off: int
    row_off: int
    width: int
    height: int

    @property
    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0

    @property
    def rio_window(self) -> rasterio.windows.Window:
        return rasterio.windows.Window(self.col_off, self.row_off, self.width, self.height)

    @property
    def affine(self) -> Affine:
        return self.grid.affine @ Affine.translation(self.col_off, self.row_off)

    @property
    def bounds(self):
        a = self.affine
        x0, y0 = a @ (0, 0)
        x1, y1 = a @ (self.width, self.height)
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def cell_indices(self, x, y):
        """Local (i, j) of the cells containing points x, y; -1 if outside."""
        a, _, c, _, e, f = tuple(self.affine)[:6]
        i = np.floor((np.asarray(x) - c) / a).astype(np.int64)
        j = np.floor((np.asarray(y) - f) / e).astype(np.int64)
        outside = (i < 0) | (i >= self.width) | (j < 0) | (j >= self.height)
        i[outside] = -1
        j[outside] = -1
        return i, j

    def read(self, path) -> np.ndarray:
        """Read band 1 for this window as float32, with nodata and negatives as 0."""
        if self.is_empty:
            return np.zeros((self.height, self.width), dtype=np.float32)
        with rasterio.open(path) as src:
            data = src.read(1, window=self.rio_window, boundless=False).astype(np.float32)
            nodata = src.nodata
        if nodata is not None and not np.isnan(nodata):
            data[data == nodata] = 0
        data[~np.isfinite(data) | (data < 0)] = 0
        return data

    def mask(self, geometries) -> np.ndarray:
        """Boolean array, True for cells whose centre lies inside `geometries`.

        Space-partitioning polygons such as HAZs therefore assign each cell
        to exactly one polygon.
        """
        if self.is_empty:
            return np.zeros((self.height, self.width), dtype=bool)
        return rasterio.features.geometry_mask(
            geometries,
            out_shape=(self.height, self.width),
            transform=self.affine,
            all_touched=False,
            invert=True,
        )
