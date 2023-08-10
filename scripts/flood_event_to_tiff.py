"""Postprocess event depths to GPKG (points) or TIFF (raster)

Inputs
------
- template raster TIFF as used for extent/transform indexing
- event, depth
"""
import os
import sys
from glob import glob

import geopandas
import numpy
import pandas
import rasterio
from tqdm import tqdm


def read_transform(fname):
    with rasterio.open(fname) as dataset:
        crs = dataset.crs
        ncols = dataset.width
        nrows = dataset.height
        transform = dataset.transform
    return crs, ncols, nrows, transform


def save_to_gpkg(df, slug):
    df = df.reset_index()
    if "T500_ID" in df.columns:
        zone_id = "T500_ID"
    else:
        zone_id = "op.id"
    lons, lats = transform * (df.row, df.col)
    df["geometry"] = geopandas.points_from_xy(lons, lats)
    gdf = geopandas.GeoDataFrame(df[[zone_id, "depth", "geometry"]])
    gdf.to_file(os.path.join("outputs", f"{slug}.gpkg"), driver="GPKG")


def save_to_tif(df, slug, nrows, ncols, transform):
    data = numpy.zeros((nrows, ncols))

    for cell in df.itertuples():
        data[cell.row, cell.col] = cell.depth

    with rasterio.open(
        os.path.join("outputs", f"{slug}.tif"),
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype=data.dtype,
        crs="+proj=latlong",
        transform=transform,
        compress="lzw",
    ) as dataset:
        dataset.write(data, 1)


if __name__ == "__main__":
    try:
        outputs = sys.argv[1].replace("--output=", "").split(",")
        print("Will output in formats (tiff and/or gpkg):", outputs)

        tiff_path = sys.argv[2]
        print(f"Using transform defined in {tiff_path}")
        event_paths = sys.argv[3:]
        print(f"Reading {len(event_paths)} events")
    except:
        print("ERROR: did not get expected arguments")
        print("Expected usage:")
        print(
            f"    python {os.path.basename(__file__)} --output=tiff,gpkg template.tiff event_*.parquet"
        )
        sys.exit()

    crs, ncols, nrows, transform = read_transform(tiff_path)

    for fname in tqdm(event_paths):
        df = pandas.read_parquet(fname)
        slug, _ = os.path.splitext(os.path.basename(fname))

        rows, cols = numpy.unravel_index(df.cell_index, (nrows, ncols))
        df["row"] = rows
        df["col"] = cols

        if "gpkg" in outputs:
            save_to_gpkg(df, slug)
        if "tiff" in outputs:
            save_to_tif(df, slug, nrows, ncols, transform)
