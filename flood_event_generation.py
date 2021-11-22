"""Calculate flood event exposure

Inputs
------
- River observation points (fluvial flooding)
- Precipitation observation points (surface water flooding)
- Hazard accumulation zones
- Simulated event return periods at observation points
- Return period flood maps
- Exposure points (these are user defined - here we pick a point in each cell of the return period
  flood maps which has any exposure across all return periods)

Outputs
-------
- Flood depth at each exposure point, for each event

"""
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import pygeos.creation
import rioxarray
from scipy.stats.mstats import gmean

JM_HAZ_T500_02 = gpd.read_file("inputs/event_points_hydrological_units/JM_HAZ_T500_02.shp")
# TODO consider rasterising zones, https://gdal.org/programs/gdal_rasterize.html
PrcipOPInfo = pd.read_csv("inputs/event_data/PrcipOPInfo.csv")
RiverOPInfo = pd.read_csv("inputs/event_data/RiverOPInfo.csv")
SimEventRP = pd.read_csv("inputs/event_data/SimEventRP.csv")
ObsEventInfo = pd.read_csv("inputs/event_data/ObsEventInfo.csv")
ObsEventRP = pd.read_csv("inputs/event_data/ObsEventRP.csv")
ObsTrack = pd.read_csv("inputs/event_data/ObsTrack.csv")

# TODO glob
SimEventRP_rcp26_2050s = pd.read_csv("inputs/future_event_sets/SimEventRP.rcp26_2050s.csv")
SimEventRP_rcp26_2080s = pd.read_csv("inputs/future_event_sets/SimEventRP.rcp26_2080s.csv")
SimEventRP_rcp45_2050s = pd.read_csv("inputs/future_event_sets/SimEventRP.rcp45_2050s.csv")
SimEventRP_rcp45_2080s = pd.read_csv("inputs/future_event_sets/SimEventRP.rcp45_2080s.csv")
SimEventRP_rcp85_2050s = pd.read_csv("inputs/future_event_sets/SimEventRP.rcp85_2050s.csv")
SimEventRP_rcp85_2080s = pd.read_csv("inputs/future_event_sets/SimEventRP.rcp85_2080s.csv")

# fluvial events event obs points intersect with hydrological units
geometry = pygeos.creation.points(RiverOPInfo["op.lon"].values, RiverOPInfo["op.lat"].values)
RiverOPInfo_gdf = gpd.GeoDataFrame(RiverOPInfo, crs="EPSG:4326", geometry=geometry)
RiverOPInfo_gdf_intersection = gpd.overlay(
    RiverOPInfo_gdf, JM_HAZ_T500_02, how="intersection"
)

# fluvial events subset and get geometric_mean_rp per hydrological unit
fluvial_events = pd.merge(
    SimEventRP,
    RiverOPInfo_gdf_intersection[["op.id", "op.lon", "op.lat", "T500_ID"]],
    on="op.id",
    how="right",
)  ## get only river obs points
fluvial_events_intersection_gm = (
    fluvial_events.groupby(["event.id", "T500_ID"])["rp"]
    .apply(list)
    .to_frame()
    .reset_index()
)
fluvial_events_intersection_gm["geometric_mean_rp"] = fluvial_events_intersection_gm[
    "rp"
].apply(gmean)

fluvial_events = pd.merge(
    SimEventRP_rcp26_2050s,
    RiverOPInfo_gdf_intersection[["op.id", "op.lon", "op.lat", "T500_ID"]],
    on="op.id",
    how="right",
)
fluvial_events_intersection_gm_rcp26_2050s = (
    fluvial_events.groupby(["event.id", "T500_ID"])["rp"]
    .apply(list)
    .to_frame()
    .reset_index()
)
fluvial_events_intersection_gm_rcp26_2050s[
    "geometric_mean_rp"
] = fluvial_events_intersection_gm_rcp26_2050s["rp"].apply(gmean)

# TODO repeat for all future sets

# interpolation bounds per event
dict = {
    "baseline": fluvial_events_intersection_gm,
    "rcp26_2050s": fluvial_events_intersection_gm_rcp26_2050s,
    # "rcp26_2080s": fluvial_events_intersection_gm_rcp26_2080s,
    # "rcp45_2050s": fluvial_events_intersection_gm_rcp45_2050s,
    # "rcp45_2080s": fluvial_events_intersection_gm_rcp45_2080s,
    # "rcp85_2050s": fluvial_events_intersection_gm_rcp85_2050s,
    # "rcp85_2080s": fluvial_events_intersection_gm_rcp85_2080s,
}

# fluvial flood maps grid intersect with hydrological units
if os.path.exists("outputs/JM_FLRF_gdf_intersection.csv"):
    JM_FLRF_gdf_intersection = pd.read_csv("outputs/JM_FLRF_gdf_intersection.csv")
else:
    JM_FLRF = rioxarray.open_rasterio('inputs/fluvial_raw_fld_depth/JM_FLRF_UD_Q20_RD_02.tif')  # should this be Q1500 for greatest extent?
    JM_FLRF = JM_FLRF.to_dataframe('results').reset_index()
    JM_FLRF = JM_FLRF[JM_FLRF['results']>0]
    geometry = pygeos.creation.points(JM_FLRF.x, JM_FLRF.y)
    JM_FLRF_gdf = gpd.GeoDataFrame(JM_FLRF,crs="EPSG:4326",geometry=geometry)
    JM_FLRF_gdf_intersection = JM_FLRF_gdf.sjoin(JM_HAZ_T500_02, predicate='within', how='left')
    JM_FLRF_gdf_intersection = JM_FLRF_gdf_intersection[["y", "x", "T500_ID", "T500_Type", "T1000_ID", "T1000_Type"]]
    JM_FLRF_gdf_intersection.to_csv('outputs/JM_FLRF_gdf_intersection.csv', index=False)

merge_cc = pd.read_csv("outputs/JM_FLRF_gdf_intersection.csv")[
    ["x", "y", "T500_ID"]
]

cc = "baseline"
fluvial_events_intersection_gm = dict[cc]
fluvial_events_intersection_gm["geometric_mean_rp"] = pd.to_numeric(
    fluvial_events_intersection_gm["geometric_mean_rp"]
)

fluvial_events_intersection_gm["interpolate_between_min_event"] = np.where(
    fluvial_events_intersection_gm["geometric_mean_rp"] <= 20,
    2,
    np.where(
        (fluvial_events_intersection_gm["geometric_mean_rp"] > 20)
        & (fluvial_events_intersection_gm["geometric_mean_rp"] <= 50),
        20,
        np.where(
            (fluvial_events_intersection_gm["geometric_mean_rp"] > 50)
            & (fluvial_events_intersection_gm["geometric_mean_rp"] <= 100),
            50,
            np.where(
                (fluvial_events_intersection_gm["geometric_mean_rp"] > 100)
                & (fluvial_events_intersection_gm["geometric_mean_rp"] <= 200),
                100,
                np.where(
                    (fluvial_events_intersection_gm["geometric_mean_rp"] > 200)
                    & (fluvial_events_intersection_gm["geometric_mean_rp"] <= 500),
                    200,
                    np.where(
                        (fluvial_events_intersection_gm["geometric_mean_rp"] > 500)
                        & (
                            fluvial_events_intersection_gm["geometric_mean_rp"]
                            <= 1500
                        ),
                        500,
                        1500,
                    ),
                ),
            ),
        ),
    ),
)
fluvial_events_intersection_gm["interpolate_between_max_event"] = np.where(
    fluvial_events_intersection_gm["geometric_mean_rp"] <= 20,
    20,
    np.where(
        (fluvial_events_intersection_gm["geometric_mean_rp"] > 20)
        & (fluvial_events_intersection_gm["geometric_mean_rp"] <= 50),
        50,
        np.where(
            (fluvial_events_intersection_gm["geometric_mean_rp"] > 50)
            & (fluvial_events_intersection_gm["geometric_mean_rp"] <= 100),
            100,
            np.where(
                (fluvial_events_intersection_gm["geometric_mean_rp"] > 100)
                & (fluvial_events_intersection_gm["geometric_mean_rp"] <= 200),
                200,
                np.where(
                    (fluvial_events_intersection_gm["geometric_mean_rp"] > 200)
                    & (fluvial_events_intersection_gm["geometric_mean_rp"] <= 500),
                    500,
                    np.where(
                        (fluvial_events_intersection_gm["geometric_mean_rp"] > 500)
                        & (
                            fluvial_events_intersection_gm["geometric_mean_rp"]
                            <= 1500
                        ),
                        1500,
                        1500,
                    ),
                ),
            ),
        ),
    ),
)

JM_FLRF_gdf_intersection = pd.read_csv("outputs/JM_FLRF_gdf_intersection.csv")
JM_FLRF_gdf_intersection["x"] = pd.to_numeric(JM_FLRF_gdf_intersection["x"]).round(10)
JM_FLRF_gdf_intersection["y"] = pd.to_numeric(JM_FLRF_gdf_intersection["y"]).round(10)

# join fluvial flood maps
for i in ["20", "50", "100", "200", "500", "1500"]:
    colname = f"rp{i}"
    JM_FLRF = rioxarray.open_rasterio(
        "inputs/fluvial_raw_fld_depth/JM_FLRF_UD_Q" + i + "_RD_02.tif"
    )
    JM_FLRF = JM_FLRF.to_dataframe("results").reset_index()
    JM_FLRF = JM_FLRF.rename(columns={"results": colname})
    JM_FLRF = JM_FLRF[JM_FLRF[colname] > 0]
    JM_FLRF["x"] = pd.to_numeric(JM_FLRF["x"]).round(10)
    JM_FLRF["y"] = pd.to_numeric(JM_FLRF["y"]).round(10)
    JM_FLRF_gdf_intersection = pd.merge(
        JM_FLRF_gdf_intersection, JM_FLRF[["x", "y", colname]], on=["x", "y"]
    )

# assume flood depth == 0 @ rp 2
JM_FLRF_gdf_intersection["rp2"] = [0] * JM_FLRF_gdf_intersection.shape[0]

# merge events to the grid based on hydrological unit
# apply log relationship between flood depths to events in event dataset
for event in (
    fluvial_events_intersection_gm["event.id"].drop_duplicates().to_list()
):
    iter_df = fluvial_events_intersection_gm[
        fluvial_events_intersection_gm["event.id"] == event
    ]

    merge = pd.merge(
        JM_FLRF_gdf_intersection, iter_df, on="T500_ID", how="right"
    )
    for it, row in merge.iterrows():
        print(it, row)
        merge.loc[it, "event"] = float(row.geometric_mean_rp)
        merge.loc[it, "event"] = np.where(
            merge.loc[it, "event"] < 2, 2, merge.loc[it, "event"]
        )
        merge.loc[it, "min_depth"] = float(row[f"rp{row.interpolate_between_min_event}"])
        merge.loc[it, "min_depth"] = np.where(
            merge.loc[it, "min_depth"] == 0,
            0,
            np.log(float(merge.loc[it, "min_depth"])),
        )
        merge.loc[it, "max_depth"] = float(row[f"rp{row.interpolate_between_max_event}"])
        merge.loc[it, "min_event"] = float(row.interpolate_between_min_event)
        merge.loc[it, "max_event"] = float(row.interpolate_between_max_event)
        merge.loc[it, "extrapolated_depth"] = merge.loc[it, "min_depth"] + (
            (merge.loc[it, "max_event"] - merge.loc[it, "min_event"])
            / (merge.loc[it, "max_event"] - merge.loc[it, "min_event"])
        ) * (merge.loc[it, "max_depth"] - merge.loc[it, "min_depth"])

    merge = merge[["x", "y", "extrapolated_depth"]]
    merge.to_csv("outputs/" + str(event) + str(cc) + ".csv")


# pluvial event obs points split using nearest neighbour
geometry = pygeos.creation.points(PrcipOPInfo["op.lon"], PrcipOPInfo["op.lat"])
PrcipOPInfo_gdf = gpd.GeoDataFrame(PrcipOPInfo, crs="EPSG:4326", geometry=geometry)

# link surface water events to observation points
surface_water_events = pd.merge(
    SimEventRP,
    PrcipOPInfo_gdf[["op.id", "op.lon", "op.lat"]],
    on="op.id",
    how="right",
)  ## get only river obs points

surface_water_events_rcp26_2050s = pd.merge(
    SimEventRP_rcp26_2050s,
    PrcipOPInfo_gdf[["op.id", "op.lon", "op.lat"]],
    on="op.id",
    how="right",
)
# TODO repeat for all future scenarios

# # interpolation bounds per event
cc_dict = {
    "baseline": surface_water_events,
    "rcp26_2050s": surface_water_events_rcp26_2050s,
    # "rcp26_2080s": surface_water_events_rcp26_2080s,
    # "rcp45_2050s": surface_water_events_rcp45_2050s,
    # "rcp45_2080s": surface_water_events_rcp45_2080s,
    # "rcp85_2050s": surface_water_events_rcp85_2050s,
    # "rcp85_2080s": surface_water_events_rcp85_2080s,
}

surface_water_events = cc_dict[cc]
surface_water_events["rp"] = pd.to_numeric(surface_water_events["rp"])

# interpolation bounds per event
surface_water_events["interpolate_between_min_event"] = np.where(
    surface_water_events["rp"] <= 20,
    2,
    np.where(
        (surface_water_events["rp"] > 20) & (surface_water_events["rp"] <= 50),
        20,
        np.where(
            (surface_water_events["rp"] > 50) & (surface_water_events["rp"] <= 100),
            50,
            np.where(
                (surface_water_events["rp"] > 100)
                & (surface_water_events["rp"] <= 200),
                100,
                np.where(
                    (surface_water_events["rp"] > 200)
                    & (surface_water_events["rp"] <= 500),
                    200,
                    np.where(surface_water_events["rp"] > 500, 500, "nan"),
                ),
            ),
        ),
    ),
)
surface_water_events["interpolate_between_max_event"] = np.where(
    surface_water_events["rp"] <= 20,
    20,
    np.where(
        (surface_water_events["rp"] > 20) & (surface_water_events["rp"] <= 50),
        50,
        np.where(
            (surface_water_events["rp"] > 50) & (surface_water_events["rp"] <= 100),
            100,
            np.where(
                (surface_water_events["rp"] > 100)
                & (surface_water_events["rp"] <= 200),
                200,
                np.where(
                    (surface_water_events["rp"] > 200)
                    & (surface_water_events["rp"] <= 500),
                    500,
                    np.where(surface_water_events["rp"] > 500, 1500, "nan"),
                ),
            ),
        ),
    ),
)

if os.path.exists("outputs/JM_FLSW_gdf_intersection.csv"):
    JM_FLSW_gdf_intersection_SW = pd.read_csv("outputs/JM_FLSW_gdf_intersection.csv")
else:
    JM_FLSW = rioxarray.open_rasterio('inputs/surface_water_raw_fld_depth/JM_FLSW_UD_Q20_RD_02.tif')
    JM_FLSW = JM_FLSW.to_dataframe('results').reset_index()
    JM_FLSW = JM_FLSW[JM_FLSW['results']>0]
    geometry = pygeos.creation.points(JM_FLSW['x'], JM_FLSW['y'])
    JM_FLSW_gdf = gpd.GeoDataFrame(JM_FLSW,crs="EPSG:4326",geometry=geometry)
    JM_FLSW_gdf_intersection_SW = JM_FLSW_gdf.sjoin_nearest(PrcipOPInfo_gdf, how='left')
    JM_FLSW_gdf_intersection_SW.to_csv('outputs/JM_FLSW_gdf_intersection.csv')

# join surface water flood maps
for i in ["20", "50", "100", "200", "500", "1500"]:
    colname = f"rp{i}"
    JM_FLSW = rioxarray.open_rasterio(
        "inputs/surface_water_raw_fld_depth/JM_FLSW_UD_Q" + i + "_RD_02.tif"
    )
    JM_FLSW = JM_FLSW.to_dataframe("results").reset_index()
    JM_FLSW = JM_FLSW.rename(columns={"results": colname})
    JM_FLSW["x"] = pd.to_numeric(JM_FLSW["x"]).round(10)
    JM_FLSW["y"] = pd.to_numeric(JM_FLSW["y"]).round(10)
    JM_FLSW = JM_FLSW[JM_FLSW[colname] > 0]
    JM_FLSW_gdf_intersection_SW = pd.merge(
        JM_FLSW_gdf_intersection_SW,
        JM_FLSW[["x", "y", colname]],
        on=["x", "y"],
        how="right",
    )

# assume flood depth == 0 @ rp 2
JM_FLSW_gdf_intersection_SW["2"] = [0] * JM_FLSW_gdf_intersection_SW.shape[0]

# merge events to the grid based on hydrological unit
# apply log relationship between flood depths to events in event dataset
for event in surface_water_events["event.id"].drop_duplicates().to_list()[1:5]:
    print(surface_water_events[surface_water_events["event.id"] == event]["op.id"])
    iter_df = surface_water_events[surface_water_events["event.id"] == event]
    merge = pd.merge(
        JM_FLSW_gdf_intersection_SW, iter_df, on="op.id", how="right"
    ).reset_index(drop=True)

    for it, row in merge.iterrows():
        print(it)
        merge.loc[it, "event"] = float(row.rp)
        merge.loc[it, "event"] = np.where(
            merge.loc[it, "event"] < 2, 2, merge.loc[it, "event"]
        )
        merge.loc[it, "min_depth"] = float(row[f"rp{row.interpolate_between_min_event}"])
        merge.loc[it, "min_depth"] = np.where(
            merge.loc[it, "min_depth"] == 0,
            0,
            np.log(float(merge.loc[it, "min_depth"])),
        )
        merge.loc[it, "max_depth"] = float(row[f"rp{row.interpolate_between_max_event}"])
        merge.loc[it, "min_event"] = float(row.interpolate_between_min_event)
        merge.loc[it, "max_event"] = float(row.interpolate_between_max_event)
        merge.loc[it, "extrapolated_depth"] = merge.loc[it, "min_depth"] + (
            (merge.loc[it, "max_event"] - merge.loc[it, "min_event"])
            / (merge.loc[it, "max_event"] - merge.loc[it, "min_event"])
        ) * (merge.loc[it, "max_depth"] - merge.loc[it, "min_depth"])
        print(it)
    merge.to_csv("outputs/" + str(event) + str(event) + "_SW.csv")
    merge_sub = merge[["x", "y", "extrapolated_depth"]]
    merge_sub = merge_sub.rename(
        columns={
            "event": event,
            "extrapolated_depth": "extrapolated_depth" + event + cc,
        }
    )
    merge_cc_SW = merge_cc_SW.merge(merge_sub, on=["x", "y"])

for rp in [5, 20, 50, 100, 200]:
    for it, row in merge_cc_SW.iterrows():
        cols_of_interest = (
            fluvial_events_intersection_gm["event.id"].drop_duplicates().to_list()
        )
        event_rp_diff = pd.DataFrame({})
        for col in cols_of_interest:
            diff = row[col] - rp
            event_rp_diff = event_rp_diff.append(
                pd.DataFrame({"rp": rp, "diff": abs(diff), "col": col}, index=[0])
            )

    event_match = event_rp_diff[
        event_rp_diff["diff"] == event_rp_diff["diff"].min()
    ]["col"].values[0]
    merge_cc_SW.loc[it, "rp_depth" + rp] = row[
        "extrapolated_depth" + event_match + cc
    ]
    merge_cc_SW.to_csv("outputs/flood_" + cc + rp + cc + ".csv")
