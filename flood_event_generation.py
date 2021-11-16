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
import numpy as np
import pandas as pd
import geopandas as gpd
import pygeos.creation
import rioxarray
from scipy.stats.mstats import gmean

JM_HAZ_T500_02 = gpd.read_file("event_points_hydrological_units/JM_HAZ_T500_02.shp")
PrcipOPInfo = pd.read_csv("event_data/PrcipOPInfo.csv")
RiverOPInfo = pd.read_csv("event_data/RiverOPInfo.csv")
SimEventRP = pd.read_csv("event_data/SimEventRP.csv")

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
    how="left",
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

# interpolation bounds per event
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
                        & (fluvial_events_intersection_gm["geometric_mean_rp"] <= 1500),
                        500,
                        "nan",
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
                        & (fluvial_events_intersection_gm["geometric_mean_rp"] <= 1500),
                        1500,
                        "nan",
                    ),
                ),
            ),
        ),
    ),
)

# fluvial flood maps grid intersect with hydrological units
if os.path.exists("JM_FLRF_gdf_intersection.csv"):
    JM_FLRF_gdf_intersection = pd.read_csv("JM_FLRF_gdf_intersection.csv", index=False)
else:
    JM_FLRF = rioxarray.open_rasterio('fluvial_raw_fld_depth/JM_FLRF_UD_Q20_RD_02.tif')  # should this be Q1500 for greatest extent?
    JM_FLRF = JM_FLRF.to_dataframe('results').reset_index()
    JM_FLRF = JM_FLRF[JM_FLRF['results']>0]
    geometry = pygeos.creation.points(JM_FLRF.x, JM_FLRF.y)
    JM_FLRF_gdf = gpd.GeoDataFrame(JM_FLRF,crs="EPSG:4326",geometry=geometry)
    JM_FLRF_gdf_intersection = JM_FLRF_gdf.sjoin(JM_HAZ_T500_02, predicate='within', how='left')
    JM_FLRF_gdf_intersection = JM_FLRF_gdf_intersection[["y", "x", "T500_ID", "T500_Type", "T1000_ID", "T1000_Type"]]
    JM_FLRF_gdf_intersection.to_csv('JM_FLRF_gdf_intersection.csv')

# join fluvial flood maps
for i in ["20", "50", "100", "200", "500", "1500"]:
    JM_FLRF = rioxarray.open_rasterio(
        "fluvial_raw_fld_depth/JM_FLRF_UD_Q" + i + "_RD_02.tif"
    )
    JM_FLRF = JM_FLRF.to_dataframe("results").reset_index()
    JM_FLRF = JM_FLRF.rename(columns={"results": i})
    JM_FLRF = JM_FLRF[JM_FLRF[i] > 0]
    JM_FLRF_gdf_intersection = pd.merge(
        JM_FLRF_gdf_intersection, JM_FLRF[["x", "y", i]], on=["x", "y"]
    )

# assume flood depth == 0 @ rp 2
JM_FLRF_gdf_intersection["2"] = [0] * JM_FLRF_gdf_intersection.shape[0]
# JM_FLRF_gdf_intersection.to_csv('JM_FLRF_gdf_intersection_2.csv')
# JM_FLRF_gdf_intersection = pd.read_csv('JM_FLRF_gdf_intersection_2.csv')

# merge events to the grid based on hydrological unit
# apply log relationship between flood depths to events in event dataset
for event in (
    fluvial_events_intersection_gm["event.id"].drop_duplicates().to_list()[1:5]
):
    iter_df = fluvial_events_intersection_gm[
        fluvial_events_intersection_gm["event.id"] == event
    ]
    merge = pd.merge(
        JM_FLRF_gdf_intersection,
        iter_df[["event.id", "T500_ID", "geometric_mean_rp"]],
        on="T500_ID",
        how="left",
    )

    merge["event"] = [iter_df["geometric_mean_rp"].values[0]] * merge.shape[0]
    merge["min_event"] = [
        iter_df["interpolate_between_min_event"].values[0]
    ] * merge.shape[0]
    merge["max_event"] = [
        iter_df["interpolate_between_max_event"].values[0]
    ] * merge.shape[0]
    merge["min_depth"] = merge[[iter_df["interpolate_between_min_event"].values[0]]]
    merge["max_depth"] = merge[[iter_df["interpolate_between_max_event"].values[0]]]
    merge["extrapolated_depth"] = merge.apply(
        lambda row: 1
        + (np.log(int(row.event)) - np.log(int(row.min_event)))
        / (np.log(int(row.max_event)) - np.log(int(row.min_event)))
        * (int(row.max_depth) - int(row.min_depth)),
        axis=1,
    )
    print(merge["extrapolated_depth"])

    # merge['extrapolated_depth'] = merge.apply(lambda row: logFunc(row['geometric_mean_rp'],row['log_a'],row['log_b']), axis=1)
    merge = merge[["x", "y", "extrapolated_depth"]]
    merge.to_csv("merge" + str(event) + ".csv")

    # df1=merge.interpolate()
    # geometry=[Point(xyz) for xyz in zip(merge.iloc[:, 0], merge.iloc[:, 1], merge.iloc[:, 2])]
    # df3=gpd.GeoDataFrame(df1, geometry=geometry)
    # df3.plot()
    # plt.savefig(str(event)+'.tiff')

# pluvial event obs points split using nearest neighbour
geometry = pygeos.creation.points(PrcipOPInfo["op.lon"], PrcipOPInfo["op.lat"])
PrcipOPInfo_gdf = gpd.GeoDataFrame(PrcipOPInfo, crs="EPSG:4326", geometry=geometry)

# link surface water events to observation points
surface_water_events = pd.merge(
    SimEventRP,
    PrcipOPInfo_gdf[["op.id", "op.lon", "op.lat"]],
    on="op.id",
    how="left",
)  ## get only river obs points

# interpolation bounds per event
surface_water_events["interpolate_between_min_event"] = np.where(
    surface_water_events["geometric_mean_rp"] <= 20,
    2,
    np.where(
        (surface_water_events["geometric_mean_rp"] > 20)
        & (surface_water_events["geometric_mean_rp"] <= 50),
        20,
        np.where(
            (surface_water_events["geometric_mean_rp"] > 50)
            & (surface_water_events["geometric_mean_rp"] <= 100),
            50,
            np.where(
                (surface_water_events["geometric_mean_rp"] > 100)
                & (surface_water_events["geometric_mean_rp"] <= 200),
                100,
                np.where(
                    (surface_water_events["geometric_mean_rp"] > 200)
                    & (surface_water_events["geometric_mean_rp"] <= 500),
                    200,
                    np.where(
                        (surface_water_events["geometric_mean_rp"] > 500)
                        & (surface_water_events["geometric_mean_rp"] <= 1500),
                        500,
                        "nan",
                    ),
                ),
            ),
        ),
    ),
)
surface_water_events["interpolate_between_max_event"] = np.where(
    surface_water_events["geometric_mean_rp"] <= 20,
    20,
    np.where(
        (surface_water_events["geometric_mean_rp"] > 20)
        & (surface_water_events["geometric_mean_rp"] <= 50),
        50,
        np.where(
            (surface_water_events["geometric_mean_rp"] > 50)
            & (surface_water_events["geometric_mean_rp"] <= 100),
            100,
            np.where(
                (surface_water_events["geometric_mean_rp"] > 100)
                & (surface_water_events["geometric_mean_rp"] <= 200),
                200,
                np.where(
                    (surface_water_events["geometric_mean_rp"] > 200)
                    & (surface_water_events["geometric_mean_rp"] <= 500),
                    500,
                    np.where(
                        (surface_water_events["geometric_mean_rp"] > 500)
                        & (surface_water_events["geometric_mean_rp"] <= 1500),
                        1500,
                        "nan",
                    ),
                ),
            ),
        ),
    ),
)

if os.path.exists("JM_FLRF_gdf_intersection_SW.csv"):
    JM_FLRF_gdf_intersection_SW = pd.read_csv("JM_FLRF_gdf_intersection_SW.csv")
else:
    JM_FLRF = rioxarray.open_rasterio('surface_water_raw_fld_depth/JM_FLSW_UD_Q20_RD_02.tif')
    JM_FLRF = JM_FLRF.to_dataframe('results').reset_index()
    JM_FLRF = JM_FLRF[JM_FLRF['results']>0]
    geometry = pygeos.creation.points(JM_FLRF['x'], JM_FLRF['y'])
    JM_FLRF_gdf = gpd.GeoDataFrame(JM_FLRF,crs="EPSG:4326",geometry=geometry)
    JM_FLRF_gdf_intersection = JM_FLRF_gdf.sjoin_nearest(PrcipOPInfo_gdf, how='left')
    JM_FLRF_gdf_intersection.to_csv('JM_FLRF_gdf_intersection_SW.csv')

# join surface water flood maps
for i in ["20", "50", "100", "200", "500", "1500"]:
    JM_FLSW = rioxarray.open_rasterio(
        "surface_water_raw_fld_depth/JM_FLSW_UD_Q" + i + "_RD_02.tif"
    )
    JM_FLSW = JM_FLSW.to_dataframe("results").reset_index()
    JM_FLSW = JM_FLSW.rename(columns={"results": i})
    print(JM_FLSW)
    JM_FLSW = JM_FLSW[JM_FLSW[i] > 0]
    JM_FLRF_gdf_intersection_SW = pd.merge(
        JM_FLRF_gdf_intersection_SW, JM_FLSW[["x", "y", i]], on=["x", "y"]
    )

# assume flood depth == 0 @ rp 2
JM_FLSW_gdf_intersection_SW["2"] = [0] * JM_FLSW_gdf_intersection_SW.shape[0]
# JM_FLSW_gdf_intersection_SW.to_csv('JM_FLSW_gdf_intersection_SW_2.csv')
# JM_FLSW_gdf_intersection_SW = pd.read_csv('JM_FLSW_gdf_intersection_SW_2.csv')

# merge events to the grid based on hydrological unit
# apply log relationship between flood depths to events in event dataset
for event in surface_water_events["event.id"].drop_duplicates().to_list()[1:5]:
    iter_df = surface_water_events[surface_water_events["event.id"] == event]
    merge = pd.merge(
        JM_FLSW_gdf_intersection_SW,
        iter_df[["event.id", "op.id", "rp"]],
        on="op.id",
        how="left",
    )

    merge["event"] = [iter_df["rp"].values[0]] * merge.shape[0]
    merge["min_event"] = [
        iter_df["interpolate_between_min_event"].values[0]
    ] * merge.shape[0]
    merge["max_event"] = [
        iter_df["interpolate_between_max_event"].values[0]
    ] * merge.shape[0]
    merge["min_depth"] = merge[[iter_df["interpolate_between_min_event"].values[0]]]
    merge["max_depth"] = merge[[iter_df["interpolate_between_max_event"].values[0]]]
    merge["extrapolated_depth"] = merge.apply(
        lambda row: 1
        + (np.log(int(row.event)) - np.log(int(row.min_event)))
        / (np.log(int(row.max_event)) - np.log(int(row.min_event)))
        * (int(row.max_depth) - int(row.min_depth)),
        axis=1,
    )
    print(merge["extrapolated_depth"])

    # merge['extrapolated_depth'] = merge.apply(lambda row: logFunc(row['geometric_mean_rp'],row['log_a'],row['log_b']), axis=1)
    merge = merge[["x", "y", "extrapolated_depth"]]
    merge.to_csv("merge" + str(event) + "_SW.csv")

    # df1=merge.interpolate()
    # geometry=[Point(xyz) for xyz in zip(merge.iloc[:, 0], merge.iloc[:, 1], merge.iloc[:, 2])]
    # df3=gpd.GeoDataFrame(df1, geometry=geometry)
    # df3.plot()
    # plt.savefig(str(event)+'.tiff')
