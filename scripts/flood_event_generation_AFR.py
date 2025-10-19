#!/usr/bin/env python
# coding: utf-8
"""Find flood event depths

Inputs
------
- River observation points (fluvial flooding)
- Precipitation observation points (surface water flooding)
- Hazard accumulation zones
- Event return periods at observation points, for each scenario (observed,
  present, future RCPs)
- Return period flood maps
- Exposure points (these can be user defined - here we pick a point in each cell
  of the return period flood maps which has any exposure across all return
  periods)

Outputs
-------
- Flood depth at each exposure point, for each event, for each scenario
- Fixed return period maps for future scenarios derived from events
"""
import logging
import os
import re
import sys
import warnings

from glob import glob
from functools import partial
from pathlib import Path
from rasterio.plot import show

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rioxarray
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import openpyxl
from scipy.stats.mstats import gmean
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import matplotlib.pyplot as plt
import seaborn as sns

# ignore warnings about sjoin_nearest with non-projected CRS
warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")


# Set of return periods, plus artificial lower/upper bound
RPS = np.array([1e-3, 2, 20, 50, 100, 200, 500, 1500, 1e6])


def main(event_set_path, data_dir):

    # Hydrological Accumulation Zones - for river flooding
    # 'T500_ID', 'T500_Type', 'T1000_ID', 'T1000_Type', 'Country', 'Area_km2',
    # 'geometry'

    country_codes = pd.read_excel(data_dir / 'incoming_data' / 'country_codes.xlsx')
    processed_data_path = data_dir / "processed_data"

    for index, row in country_codes.iterrows():
        
        i = row['HAZ']
        j = row['ISO_3166_alpha2']
        k = row['ISO_3166_alpha3']
        # k = row['Region'] REGION IS SPECIFIED IN THE EVENT SET FOLDER - GIVEN AS ARGUMENT OF THE SCRIPT
        m = row['Country_undefended']
        path = data_dir / f"incoming_data/HAZ/JBA_HAZ_{i}_02/{j}_HAZ_T500_02.shp"
        print(f"Trying to read: {path}")
        if not os.path.exists(path):
            print(f"File does not exist: {path}")
            continue
        hydrological_accumulation_zones = gpd.read_file(path)[["T500_ID", "geometry"]]
        print(hydrological_accumulation_zones.head())

        # River Observation Points (OP)
        # 'op.id', 'op.lon', 'op.lat', 'region', 'sub.region', 'agg.zone',
        # 'catchment.area', 'cent.lon', 'cent.lat'
        # Region 11 and 12 for Africa

        river_ops = latlon_to_gdf(
            pd.read_csv(data_dir / "incoming_data" / "GlobalEventSet" / "R11" / "RiverOpInfo_20161207.csv"),
            lat_column="op.lat",
            lon_column="op.lon",
        )[["op.id", "geometry"]]
        river_haz_op = link_haz_op(hydrological_accumulation_zones, river_ops)
        river_rp_points = read_rp_maps_to_points(str(
            data_dir / "incoming_data" / "GlobalUndefendedFloodMaps" / m / "WGS84" / "GeoTIFFs" / f"GFM_{m}_2016_WGS84_RD_River" / f"{j}_FLRF_UD_*_RD_02.tif"
        ))

        # (T500_ID) cell_index, rp100, rp1500, rp200, rp20, rp500, rp50, rp2, geometry

        river_exposure_points = link_haz_ep(
            hydrological_accumulation_zones, river_rp_points
        )

        # Event-OP return period
        # 'event.id', 'op.id', 'rp', 'peak.day.id', 'start.day.id', 'end.day.id'
        #
        
        # Files:
        # - "inputs/event_data/ObsEventRP.csv"
        # - "inputs/event_data/SimEventRP.csv"
        
        
        # Read, set index and keep only return period column
        # (op.id, event.id) rp

        event_set = pd.read_csv(
            event_set_path, usecols=["event.id", "op.id", "rp"]
        ).set_index(["op.id", "event.id"])
        scenario_prefix = os.path.splitext(os.path.basename(event_set_path))[0]

        # Calculate river event exposure
        output_dir = Path(processed_data_path / "outputs" / f"{scenario_prefix}_{j}")
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Processing river events.")
        river_events = link_event_op_haz(event_set, river_haz_op)
        river_events.to_csv(processed_data_path / "outputs" / f"{scenario_prefix}_{j}_river.csv")
        interpolate_event_exposure(
            river_events,
            river_exposure_points,
            hazard_prefix="FLRF",
            output_dir=output_dir,
        )
        
        # -------------------------------------------------------------------------------------------------------------------#
        ## STAT AND VISUAL ANALYSIS ##
        # -------------------------------------------------------------------------------------------------------------------#

        river_event_years = pd.read_csv(data_dir / "incoming_data" / "GlobalEventSet" / "R11" / "ObsEventInfo_20161207.csv")
        subset_years = river_event_years[
        (river_event_years[k] == "TRUE") & 
        (river_event_years["is.river"] == "TRUE")
        ]

        re_haz_rp = pd.read_csv(processed_data_path / "outputs" / f"{scenario_prefix}_{j}_river.csv")

        # For each event and area, compute mean, 90th percentile, 10th percentile of depth
        CELL_SIZE_M = 30   # <-- APPROXIMATE, SHOULD BE READ FROM RASTER METADATA
        CELL_AREA = CELL_SIZE_M * CELL_SIZE_M

        def compute_stats(row):
            folder_path = os.path.join(
                processed_data_path,
                "outputs",
                f"{scenario_prefix}_{j}",
                f"T500_ID={row['T500_ID']}",
                f"event={row['event.id']}"
            )
            if not os.path.exists(folder_path):
                return pd.Series([0.0, 0.0, 0.0, 0.0])
            
            # find parquet files inside this folder
            parquet_files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
            if not parquet_files:
                return pd.Series([np.nan, np.nan, np.nan, np.nan])
            
            # read them all
            dfs = [pd.read_parquet(os.path.join(folder_path, f)) for f in parquet_files]
            data = pd.concat(dfs, ignore_index=True)
            
            if "depth" not in data.columns:
                return pd.Series([np.nan, np.nan, np.nan, np.nan])
            
            depth = data["depth"].dropna()

            # compute flooded area (cells with depth > 0.0)
            flooded_cells = (depth > 0).sum()
            flooded_area = flooded_cells * CELL_AREA
            
            return pd.Series([
                depth.mean(),
                np.median(depth),
                np.percentile(depth, 90),
                np.percentile(depth, 10),
                flooded_area
            ])
        

            
        # Add new columns with stats
        re_haz_rp[["depth_avg", "depth_med", "depth_p90", "depth_p10", "flooded_area_m2"]] = \
            re_haz_rp.apply(compute_stats, axis=1)

        figure_folder = data_dir/"figures" / f"{scenario_prefix}_{j}"
        os.makedirs(figure_folder, exist_ok=True)

        # Depth stats across events (already in re_haz_rp)
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=re_haz_rp, x="T500_ID", y="depth_avg")
        plt.title("Distribution of average depth across events by area")
        plt.xlabel("T500_ID")
        plt.ylabel("Average Depth (m)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, f"boxplot_depth_avg_area.png"))
        plt.close()
        
        # Depth stats by event across areas (already in re_haz_rp)
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=re_haz_rp, x="event.id", y="depth_avg")
        plt.title("Distribution of average depth across areas by event")
        plt.xlabel("Event ID")
        plt.ylabel("Average Depth (m)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, f"boxplot_depth_avg_event.png"))
        plt.close()

        

        ## Return period histogram by area ##

        for area in re_haz_rp["T500_ID"].unique():
            subset_area = re_haz_rp[re_haz_rp["T500_ID"] == area]
            
            # Define the folder path for this T500_ID
            base_folder = processed_data_path / "outputs" / f"{scenario_prefix}_{j}" / f"T500_ID={area}"
            hist_folder = os.path.join(base_folder, "analysis")
            os.makedirs(hist_folder, exist_ok=True)

            # Histogram of rp
            plt.figure(figsize=(12, 12))
            counts, bins, patches = plt.hist(
                subset_area['rp'],
                bins=20,
                weights=[100/len(subset_area)]*len(subset_area),  
                edgecolor='black'
            )
            plt.title(f"Histogram of rp for T500_ID: {area}")
            plt.xlabel("rp")
            plt.ylabel("Frequency (%)")
            plt.savefig(os.path.join(hist_folder, f"histogram_rp_T500_ID_{area}.png"))
            plt.close()

            print(f"Histogram saved for T500_ID={area}")


        ## Map of average flooded area for each HAZ ##

        haz_shapefile = gpd.read_file(path)
        gdf = re_haz_rp.merge(
            haz_shapefile[["T500_ID", "geometry"]],
            on="T500_ID",
            how="left"
        )
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=haz_shapefile.crs)

        metrics = {
            "flooded_area_km2": "Average Flooded Area (km²)",
        }

        # Convert flooded area to km2
        gdf["flooded_area_km2"] = gdf["flooded_area_m2"] / 1e6

        base = haz_shapefile.copy()
        print(gdf.columns)

        for ev in re_haz_rp["event.id"].unique():
            subset = gdf[gdf["event.id"] == ev]

            for col, title in metrics.items():
                fig, ax = plt.subplots(figsize=(10, 8))

                # Draw grey background (all polygons)
                base.plot(color="lightgrey", edgecolor="black", linewidth=0.3, ax=ax)

                # Draw event polygons with color scale (in km2)
                subset.plot(
                    column=col,
                    cmap="Blues",
                    linewidth=0.3,
                    edgecolor="black",
                    legend=True,
                    ax=ax
                )

                ax.set_title(f"{title}\n(Event={ev})", fontsize=12)
                ax.axis("off")

                outpath = os.path.join(figure_folder, f"map_{col}_event_{ev}.png")
                plt.savefig(outpath, bbox_inches="tight", dpi=300)
                plt.close()

        print(f"Maps saved in {figure_folder}")

        # --- Boxplots by event and by year of the depths --- #
        all_event_dfs = []

        for ev1 in re_haz_rp["event.id"].unique(): 
            for area in re_haz_rp["T500_ID"].unique():
                folder_path = os.path.join(
                    processed_data_path,
                    "outputs",
                    f"{scenario_prefix}_{j}",
                    f"T500_ID={area}",
                    f"event={ev1}"
                )
            
                if not os.path.exists(folder_path):
                    continue
                
                parquet_files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
                if not parquet_files:
                    continue
                
                tiff_path = data_dir / "incoming_data" / "GlobalUndefendedFloodMaps" / m / "WGS84" / "GeoTIFFs" / f"GFM_{m}_2016_WGS84_RD_River" / f"{j}_FLRF_UD_Q100_RD_02.tif"
                crs, ncols, nrows, transform = read_transform(tiff_path)

                
                dfs = []
                
                for f in parquet_files:
                    df = pd.read_parquet(os.path.join(folder_path, f),)
                    rows, cols = np.unravel_index(df.cell_index, (nrows, ncols))
                    df["row"] = rows
                    df["col"] = cols

                    pixel_width = transform.a
                    pixel_height = abs(transform.e)  # usually negative
                    
                    # Initialize area column with zeros
                    df["flood_area_km2"] = 0.0
                    
                    # Only calculate area where depth > 0
                    flooded_mask = df["depth"] > 0
                    
                    
                    pixel_area_km2 = (pixel_width * pixel_height) / 1e6
                    df.loc[flooded_mask, "flood_area_km2"] = pixel_area_km2

                    slug = os.path.join(folder_path, f"{ev1}_{area}.tif")
                    
                    save_to_tif(df, slug, nrows, ncols, transform)
                    
                    # out_png = os.path.join(figure_folder, f"{ev1}_{area}.png")
                    # plot_tif_with_shapefile(slug, path, out_png)
                    
                    df["event.id"] = ev1
                    
                    df["T500_ID"] = area
                    
                    # Add source file column
                    df["source_file"] = f
                    dfs.append(df)
                
                if dfs:
                    all_event_dfs.append(pd.concat(dfs, ignore_index=True))

                         
               
                

        # Merge all events together
        all_event_depths = pd.concat(all_event_dfs, ignore_index=True)

        # Keep only what's needed
        all_event_depths = all_event_depths[["event.id", "depth","T500_ID","flood_area_km2"]].dropna()
        #all_event_depths = all_event_depths.merge(re_haz_rp,on=["event.id","T500_ID"])
        
        all_event_depths.to_csv(processed_data_path / "outputs" / f"{scenario_prefix}_{j}_river_with_depths.csv")
        
        # merge with years
        merged = all_event_depths.merge(
            subset_years[["event.id", "start.year"]],
            on="event.id",
            how="left"
        )

        merged.to_csv(processed_data_path / "outputs" / f"{scenario_prefix}_{j}_river_with_depths_withyears.csv")

        # --- Boxplot across ALL events ---
        plt.figure(figsize=(18, 8))
        sns.boxplot(data=all_event_depths, x="event.id", y="depth")
        plt.title("Distribution of Depths Across Events")
        plt.xlabel("Event ID")
        plt.ylabel("Depth (m)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, "boxplot_depth_byevents.png"), dpi=300)
        plt.close()

        
        # Boxplot by year (all events pooled) ---
        if "start.year" in merged.columns and merged["start.year"].notna().any():
            plt.figure(figsize=(14, 8))
            sns.boxplot(data=merged, x="start.year", y="depth")
            plt.title("Distribution of Depths by Year (All Events)")
            plt.xlabel("Year")
            plt.ylabel("Depth (m)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(figure_folder, "boxplot_depth_byyear.png"), dpi=300)
            plt.close()
        
        # -------------------------------------------------------
        # Compute summary statistics per event
        # -------------------------------------------------------
        event_stats = (
            all_event_depths.groupby("event.id")
            .agg(
                mean_depth=("depth", "mean"),
                max_depth=("depth", "max"),
                median_depth=("depth", "median"),
                flooded_area=("flood_area_km2", "sum"),  
                n_basins=("T500_ID", pd.Series.nunique),
            )
            .reset_index()
        )

        print("Summary statistics per event computed:")
        print(event_stats.head())

        # -------------------------------------------------------
        # Save event-level stats
        # -------------------------------------------------------
        

        event_stats_outfile = os.path.join(output_dir, "event_statistics.csv")
        event_stats.to_csv(event_stats_outfile, index=False)
        print(f"Saved event statistics to {event_stats_outfile}")

        # -------------------------------------------------------
        # Count how many events affected each catchment
        # -------------------------------------------------------
        catchment_event_counts = (
            all_event_depths.groupby("T500_ID")["event.id"]
            .nunique()
            .reset_index(name="n_events")
        )

        print("Count of events per catchment computed:")
        print(catchment_event_counts.head())

        # -------------------------------------------------------
        # Merge with basin shapefile
        # -------------------------------------------------------
        print("Inspecting shapefile columns...")
        print(gdf.columns)

        if "T500_ID" not in gdf.columns:
            raise ValueError("Shapefile must contain a 'T500_ID' column to join with event counts")

        basins_with_counts = gdf.merge(catchment_event_counts, on="T500_ID", how="left")
        basins_with_counts["n_events"] = basins_with_counts["n_events"].fillna(0)

        # -------------------------------------------------------
        # Save merged GeoPackage
        # -------------------------------------------------------
        gpkg_outfile = os.path.join(output_dir, "basins_with_event_counts.gpkg")
        basins_with_counts.to_file(gpkg_outfile, driver="GPKG")
        print(f"Saved GeoPackage with event counts to {gpkg_outfile}")

        # -------------------------------------------------------
        # VISUALIZATIONS 
        # -------------------------------------------------------

        # Histogram of event statistics
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        sns.histplot(event_stats["mean_depth"], bins=20, ax=axes[0,0], kde=True)
        axes[0,0].set_title("Mean Flood Depth per Event")

        sns.histplot(event_stats["flooded_area"], bins=20, ax=axes[0,1], kde=True)
        axes[0,1].set_title("Total Flooded Area per Event (km²)")

        sns.histplot(event_stats["n_basins"], bins=20, ax=axes[1,0], kde=True)
        axes[1,0].set_title("Number of Basins per Event")

        sns.scatterplot(data=event_stats, x="mean_depth", y="flooded_area", ax=axes[1,1])
        axes[1,1].set_title("Mean Depth vs Flooded Area")
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, "event_statistics_histograms.png"), dpi=300)
        plt.close()
        print("Saved figure: event_statistics_histograms.png")

        # Map of event counts per basin
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        basins_with_counts.plot(column="n_events", cmap="viridis", legend=True, ax=ax)
        ax.set_title("Number of Flood Events per Basin", fontsize=14)
        ax.axis("off")
        plt.savefig(os.path.join(figure_folder, "map_event_counts_per_basin.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved figure: map_event_counts_per_basin.png")

        # Distribution of event counts per catchment
        plt.figure(figsize=(6, 4))
        sns.countplot(x="n_events", data=catchment_event_counts)
        plt.title("Distribution of Event Counts per Catchment")
        plt.xlabel("Number of Events")
        plt.ylabel("Number of Catchments")
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, "distribution_event_counts_per_catchment.png"), dpi=300)
        plt.close()
        print("Saved figure: distribution_event_counts_per_catchment.png")


       # Bar plot: Summed flooded area per event by year
        if not merged["start.year"].isna().all():
            plt.figure(figsize=(10, 6))
            
            # Group by year and event to get total flooded area per event per year
            grouped_area = (
                merged.groupby(["start.year", "event_id"], as_index=False)["flood_area_km2"]
                .sum()
            )

            # Bar plot: each event is a separate bar within its year
            sns.barplot(
                data=grouped_area,
                x="start.year",
                y="flood_area_km2",
                hue="event_id",
                dodge=True,
                palette="viridis",
                edgecolor="k"
            )

            plt.title("Summed Flooded Area per Event by Year")
            plt.xlabel("Year")
            plt.ylabel("Flooded Area (km²)")
            plt.legend(title="")
            plt.tight_layout()
            plt.savefig(os.path.join(figure_folder, "flooded_area_per_event_by_year.png"), dpi=300)
            plt.close()
            print("Saved figure: flooded_area_per_event_by_year.png")
        

def link_event_op_haz(events, haz_op):
    # Link event OPs to HAZ (drop OPs which are not linked)
    # op.id, event.id, rp, T500_ID
    events = events.reset_index().merge(haz_op, on="op.id").dropna()

    # Take the geometric mean of Event/OP return periods if multiple OPs per HAZ
    # (T500_ID, event.id) rp
    return events.groupby(["event.id", "T500_ID"]).agg({"rp": gmean})


def link_haz_op(haz, ops):
    # Link OPs to HAZs
    haz_within = haz.sjoin(ops, predicate="contains", how="right")[["T500_ID", "op.id"]]

    # some HAZ do not contain an OP
    haz_remaining = haz[~haz.T500_ID.isin(haz_within.T500_ID.unique())]
    haz_nearest = haz_remaining.sjoin_nearest(ops, how="left")[["T500_ID", "op.id"]]
    return pd.concat([haz_within, haz_nearest])


def link_haz_ep(haz, eps):
    return (
        eps.sjoin(haz, predicate="within", how="left"
        )
        .reset_index()
        .drop(columns="index_right")
        .set_index("T500_ID")
    )


def latlon_to_gdf(df, lat_column="lat", lon_column="lon"):
    geometry = gpd.points_from_xy(df[lon_column], df[lat_column])
    return gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)


def read_rp_map(fname):
    """Read flood maps, use all cells with any depth > 0 as potential exposure
    points
    """
    rp = re.search(r"Q(\d+)_", fname).group(1)
    colname = f"rp{rp}"
    with rasterio.open(fname) as dataset:
        data = dataset.read(1)
        data[data == dataset.nodata] = np.nan
        df = pd.DataFrame({colname: data.flatten()})

    df = df[df[colname] > 0].dropna()
    return df


def get_xy_df(fname):
    """Read a raster file, convert to dataframe retaining only x,y coordinate
    values  and 0..n index
    """
    xy = (
        rioxarray.open_rasterio(fname)
        .to_dataframe("data")
        .reset_index()
        .drop(columns=["band", "spatial_ref", "data"])
    )
    return xy


def read_rp_maps_to_points(pattern):
    # Define as an iter to use each file once
    rp_maps = iter(sorted(glob(pattern)))

    # Read first
    rp_points = read_rp_map(next(rp_maps))
    # Read the rest
    for fname in rp_maps:
        df = read_rp_map(fname)
        rp_points = rp_points.join(df, how="outer")

    # Fill NA with zeros
    rp_points = rp_points.fillna(0)
    rp_points["rp2"] = 0
    xy = get_xy_df(fname)
    rp_points = rp_points.join(xy)

    # name index
    rp_points.index.rename("cell_index", inplace=True)

    # Convert to GeoDataFrame
    return latlon_to_gdf(rp_points, lat_column="y", lon_column="x").drop(
        columns=["y", "x"]
    )


def interpolate_rp_factor(df):
    return (np.log(df.rp) - np.log(df.rp_l)) / (np.log(df.rp_u) - np.log(df.rp_l))


def interpolate_depth_df(df):
    depth = df.depth_l + ((df.depth_u - df.depth_l) * df.rp_factor)

    return depth


def interpolate_event_exposure(
    event_zones, exposure_points, hazard_prefix, output_dir
):
    # Cap at max RP 1500
    event_zones.loc[event_zones.rp >= 1500, "rp"] = 1500

    bin_index = np.searchsorted(RPS, event_zones.rp, side="left")
    event_zones["bin_index"] = bin_index
    event_zones["rp_l"] = RPS[bin_index - 1]
    event_zones["rp_u"] = RPS[bin_index]
    event_zones["rp_factor"] = interpolate_rp_factor(event_zones)
    # event_zones is now a dataframe with:
    # (T500_ID, event.id) rp, bin_index, rp_l, rp_u, rp_factor

    event_ids = sorted(event_zones.reset_index()["event.id"].unique())
    event_depths_partial = partial(
        event_depths,
        event_zones=event_zones,
        exposure_points=exposure_points,
        hazard_prefix=hazard_prefix,
        output_dir=output_dir,
    )
    process_map(
        event_depths_partial,
        event_ids,
        chunksize=32,
        max_workers=int(os.cpu_count() / 2),
    )


def event_depths(
    event_id, event_zones, exposure_points, hazard_prefix, output_dir
):
    # Each HAZ in this event, with RP values
    event_haz = event_zones.loc[event_id].reset_index()
    # All points for this event, joined with RP values via HAZ
    event_points = exposure_points.loc[event_haz.T500_ID].join(
        event_haz.set_index("T500_ID")
    )

    event_points.bin_index = event_points.bin_index.astype(np.int32)

    if len(event_points):
        depths = [
            0,
            event_points.rp2,
            event_points.rp20,
            event_points.rp50,
            event_points.rp100,
            event_points.rp200,
            event_points.rp500,
            event_points.rp1500,
        ]
        event_points["depth_l"] = np.choose(event_points.bin_index - 1, depths)
        event_points["depth_u"] = np.choose(event_points.bin_index, depths)
        event_points["depth"] = interpolate_depth_df(event_points)
        # Any RP < 2 gets zero depth
        event_points.loc[event_points.rp <= 2, "depth"] = 0

        # Output cells
        # T500_ID, depth, cell_index, event
        event_points = event_points.reset_index()[["T500_ID", "depth", "cell_index"]]
        event_points = event_points[event_points.depth > 0]
        event_points["event"] = event_id
        event_points["hazard"] = hazard_prefix

        event_points.to_parquet(
            output_dir, partition_cols=["T500_ID", "event"], index=False
        )

def read_transform(fname):
    with rasterio.open(fname) as dataset:
        crs = dataset.crs
        ncols = dataset.width
        nrows = dataset.height
        transform = dataset.transform
    return crs, ncols, nrows, transform

def save_to_tif(df, slug, nrows, ncols, transform):
    data = np.zeros((nrows, ncols))

    for cell in df.itertuples():
        data[cell.row, cell.col] = cell.depth

    with rasterio.open(
        slug,
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



def plot_tif_with_shapefile(tif_path, shapefile_path, out_png):
    with rasterio.open(tif_path) as src:
        raster_data = src.read(1)
        raster_transform = src.transform
        raster_crs = src.crs

    gdf = gpd.read_file(shapefile_path).to_crs(raster_crs)

    fig, ax = plt.subplots(figsize=(10, 8))

    # mask zeros so background stays transparent
    masked = np.ma.masked_where(raster_data == 0, raster_data)

    # plot raster first
    show(masked, transform=raster_transform, cmap="Blues", ax=ax, alpha=0.8)

    # then add shapefile outlines on top
    gdf.boundary.plot(ax=ax, color="black", linewidth=0.5, alpha=0.7)

    plt.title("Flood event map", fontsize=14)
    plt.axis("off")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved image: {out_png}")



if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO
    )

    try:
        event_set_path = sys.argv[1]
        data_dir = Path(sys.argv[2])
        logging.info(f"Processing events defined in {event_set_path}")
    except:
        logging.error(
            f"""Did not get expected arguments
Expected usage:
    python {os.path.basename(__file__)} event_set.csv path/to/data_dir
"""
        )
        exit()
    # data_dir = Path(r"C:/Users/cenv1075/Desktop/DataFolders/JBA_flooding/")
    main(event_set_path, data_dir)
    logging.info("Done.")
