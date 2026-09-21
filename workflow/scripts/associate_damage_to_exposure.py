import click
import logging
import os


from glob import glob

import pandas as pd
import geopandas as gpd

import numpy as np
import snail.damages as sndam


@click.command()
@click.version_option("1.0")

@click.option(
    "--depth_path",
    required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True),
    help="Path to depth file folder",
)

@click.option(
    "--asset_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
    help="Path to asset GeoParquet",
)

@click.option(
    "--cost_path",
    required=True,
    type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True),
    help="Path to costs folder containing csv files for each asset type, e.g. costs/road_edges.csv, costs/airport_field_polygons.csv, costs/iww.csv, etc.",
)

@click.option(
    "--curves_path",
    required=True,
    type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True),
    help="Path to curves folder, e.g. curves/road/, curves/airport_field/, curves/iww/, etc.",
)

@click.option(
    "--output_path",
    required=True,
    type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True),
    help="Path to damage geoparquet file",
)


def main(depth_path, asset_path, cost_path, curves_path, output_path):
    
    """
    Example usage:

        python workflow/scripts/associate_damage_to_exposure.py \
            --depth_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/ObsEvents/Undefended/road_edges/\
            --asset_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/split_road_edges_network_prova.parquet\
            --cost_path ~/Desktop/DataFolders/JBA_flooding/processed_data/costs/\
            --curves_path ~/Desktop/DataFolders/JBA_flooding/processed_data/curves/\
            --output_path ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/ObsCosts/Undefended/road_edges/
    """
    
    
    os.makedirs(output_path, exist_ok=True)
    depth_files = sorted(glob(os.path.join(depth_path, "event=*/*.parquet")))

    asset_name = os.path.basename(os.path.normpath(depth_path))  # e.g. "road_edges", "airport_field_polygons"
    asset_type = asset_name.rsplit("_", 1)[0]                         # e.g. "road", "airport_field"

    # Load assets
    asset = gpd.read_parquet(asset_path)

    epsg = 4326
    asset = asset.to_crs(epsg)

    # Determine whether this asset is measured by length or area, from geometry type
    geom_types = asset.geometry.geom_type.unique()
    if set(geom_types) <= {"LineString", "MultiLineString"}:
        unit_mode = "length"
    elif set(geom_types) <= {"Polygon", "MultiPolygon"}:
        unit_mode = "area"
    else:
        raise ValueError(f"Unexpected/mixed geometry types for {asset_name}: {geom_types}")

    # Ensure the relevant size column exists 
    if unit_mode == "length" and "length_m" not in asset.columns:
        asset["length_m"] = asset.geometry.length
    if unit_mode == "area" and "area_m2" not in asset.columns:
        asset["area_m2"] = asset.geometry.area

    size_col = "length_m" if unit_mode == "length" else "area_m2"
    cost_col_in_table = "mean_cost_usd_per_m" if unit_mode == "length" else "mean_cost_usd_per_m2"
    cost_col_out = "cost_usd_per_m" if unit_mode == "length" else "cost_usd_per_m2"

    # Cost table for this specific asset file
    cost = pd.read_csv(os.path.join(cost_path, f"{asset_name}.csv"))

    # Damage curves for this asset type: one CSV per cost_type, in curves/{asset_type}/
    curve_files = glob(os.path.join(curves_path, asset_type, "*.csv"))
    curves = {os.path.splitext(os.path.basename(f))[0]: f for f in curve_files}

    for parquet_path in depth_files:
        depth_event = pd.read_parquet(parquet_path)
        depth_event = depth_event.rename(columns={"depth": "depth_m"})

        asset_event = asset.merge(
            depth_event[["id", "depth_m"]],
            on="id",
            how="left",
        )

        # Attach unit cost by matching cost_type
        asset_event = asset_event.merge(
            cost[["cost_type", cost_col_in_table]],
            on="cost_type",
            how="left",
        ).rename(columns={cost_col_in_table: cost_col_out})

        # Compute proportion_damaged per cost_type, using the matching curve file
        asset_event["proportion_damaged"] = np.nan
        for cost_type in asset_event["cost_type"].dropna().unique():
            curve_file = curves.get(cost_type)
            if curve_file is None:
                print(f"Warning: no curve found for cost_type '{cost_type}' in curves/{asset_type}/")
                continue

            damage_curve = sndam.PiecewiseLinearDamageCurve.from_csv(
                curve_file,
                intensity_col="flood_depth_m",
                damage_col="damage_fraction_mean",
                comment="#",
            )
            mask = asset_event["cost_type"] == cost_type
            depths = asset_event.loc[mask, "depth_m"]
            asset_event.loc[mask, "proportion_damaged"] = damage_curve.damage_fraction(depths)

        asset_event["damage_usd"] = (
            asset_event[size_col] * asset_event[cost_col_out] * asset_event["proportion_damaged"]
        )

        event_name = os.path.basename(os.path.dirname(parquet_path))   # e.g. "event=E001"
        out_file = os.path.join(output_path, f"{event_name}.parquet")
        asset_event.to_parquet(out_file)
        logging.info(f"Wrote {out_file}")
    

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s",
        level=logging.INFO
    )

    main()