"""Split one asset layer on the hazard grid within one HAZ.

The script reads only a window of each continental RP map and a bbox subset
of the prepared asset layer. A piece belongs to this HAZ if the centre of its
grid cell lies inside the HAZ polygon. HAZs partition space, so every piece is
assigned to at most one HAZ.

The output has one row per piece:

    split_id, id, cost_type, raster_i, raster_j (global grid indices),
    size, size_unit (m, m2 or count), sop (standard of protection RP, NaN if
    undefended), depth_rp{RP} for each mapped RP, geometry

Example:

    python workflow/scripts/split_assets.py \
        --haz_id 500_13_19495 --haz_path processed_data/basins/haz_500.gpkg --haz_id_column T500_ID \
        --grid processed_data/prepared/hazard_grid.json \
        --rp 20 --rp_path processed_data/hazards/flrf_ud_Q20.tif ... \
        --layer road_edges --asset_path processed_data/prepared/assets/road_edges.parquet \
        --defended_path processed_data/defended_areas/defended_areas.gpkg --sop_column JBA_SoP \
        --output processed_data/exposure/road_edges/haz=500_13_19495/splits.parquet
"""

import logging

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import CRS, Transformer

from floodlib.grid import HazardGrid
from floodlib.split import layer_kind, piece_sizes, representative_points, split_on_window

SIZE_UNITS = {"line": "m", "polygon": "m2", "point": "count"}


@click.command()
@click.option("--haz_id", required=True)
@click.option("--haz_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--haz_id_column", required=True)
@click.option("--grid", "grid_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--rp", "rps", required=True, multiple=True, type=float, help="Mapped RP, repeat in order")
@click.option("--rp_path", "rp_paths", required=True, multiple=True, type=click.Path(exists=True), help="RP map for each --rp")
@click.option("--layer", required=True)
@click.option("--asset_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--defended_path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--sop_column", default="JBA_SoP", show_default=True)
@click.option("--output", required=True, type=click.Path(dir_okay=False, writable=True))
def main(haz_id, haz_path, haz_id_column, grid_path, rps, rp_paths, layer, asset_path, defended_path, sop_column, output):
    if len(rps) != len(rp_paths) or list(rps) != sorted(rps):
        raise ValueError("Give one --rp_path per --rp, with --rp in increasing order")
    kind = layer_kind(layer)
    grid = HazardGrid.from_json(grid_path)

    haz = read_haz(haz_path, haz_id_column, haz_id).to_crs(grid.crs)
    window = grid.window_for_bounds(haz.total_bounds)
    logging.info("HAZ %s window: %s", haz_id, window)

    assets = gpd.read_parquet(asset_path, columns=["id", "cost_type", "geometry"], bbox=tuple(window.bounds))
    logging.info("Read %d %s features in HAZ bbox", len(assets), layer)

    pieces = split_on_window(assets, window, kind)
    in_haz = window.mask(haz.geometry)
    if len(pieces):
        pieces = pieces[in_haz[pieces.j.values, pieces.i.values]].reset_index(drop=True)
    logging.info("%d pieces in HAZ cells", len(pieces))

    for rp, path in zip(rps, rp_paths):
        data = window.read(path)
        pieces[depth_column(rp)] = data[pieces.j.values, pieces.i.values].astype(np.float32)

    pieces["raster_i"] = (pieces.i + window.col_off).astype(np.int32)
    pieces["raster_j"] = (pieces.j + window.row_off).astype(np.int32)
    pieces["size"] = piece_sizes(pieces, kind).astype(np.float64)
    pieces["size_unit"] = SIZE_UNITS[kind]
    pieces["sop"] = piece_sop(pieces, kind, defended_path, sop_column, window.bounds, grid.crs)
    pieces["split_id"] = np.arange(len(pieces), dtype=np.int32)
    pieces["cost_type"] = pieces["cost_type"].astype(object)

    columns = ["split_id", "id", "cost_type", "raster_i", "raster_j", "size", "size_unit", "sop"]
    columns += [depth_column(rp) for rp in rps] + ["geometry"]
    pieces = gpd.GeoDataFrame(pieces[columns], geometry="geometry", crs=grid.crs)
    pieces.to_parquet(output, index=False)
    logging.info("Wrote %d pieces to %s", len(pieces), output)


def depth_column(rp: float) -> str:
    return f"depth_rp{rp:g}"


def read_haz(haz_path, haz_id_column, haz_id) -> gpd.GeoDataFrame:
    haz = gpd.read_file(haz_path, columns=[haz_id_column])
    haz = haz[haz[haz_id_column].astype(str) == str(haz_id)]
    if len(haz) == 0:
        raise ValueError(f"HAZ {haz_id} not found in {haz_path}")
    return haz


def piece_sop(pieces, kind, defended_path, sop_column, bounds, crs) -> pd.Series:
    """Highest standard of protection (as an RP) of the defended areas containing each piece."""
    sop = pd.Series(np.nan, index=pieces.index)
    if defended_path is None or pieces.empty:
        return sop
    defended_crs = CRS.from_user_input(pyogrio.read_info(defended_path)["crs"])
    to_defended = Transformer.from_crs(CRS.from_user_input(crs), defended_crs, always_xy=True)
    defended_bounds = to_defended.transform_bounds(*bounds)
    defended = gpd.read_file(defended_path, columns=[sop_column], bbox=defended_bounds)
    if defended.empty:
        return sop
    defended = defended.to_crs(crs)
    defended = defended[defended[sop_column] > 0]

    points = gpd.GeoDataFrame(geometry=representative_points(pieces, kind))
    joined = gpd.sjoin(points, defended[[sop_column, "geometry"]], predicate="within", how="inner")
    best = joined.groupby(level=0)[sop_column].max()
    sop.loc[best.index] = best.values
    logging.info("%d of %d pieces in defended areas", len(best), len(pieces))
    return sop


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
