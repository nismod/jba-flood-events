#!/usr/bin/env python
# coding: utf-8
"""Find flood event depths

Inputs
------
- River observation points (fluvial flooding)
- Precipitation observation points (surface water flooding)
- Hazard accumulation zones
- Event return periods at observation points, for each scenario (observed, present, future RCPs)
- Return period flood maps
- Exposure points (these are user defined - here we pick a point in each cell of the return period
  flood maps which has any exposure across all return periods)

Outputs
-------
- Flood depth at each exposure point, for each event, for each scenario
- Fixed return period maps for future scenarios derived from events
"""
import os
import re
import sys
import warnings
from glob import glob

# ignore warnings about geopandas parquet implementation
warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
# ignore warnings about GEOS-PyGEOS conversions
warnings.filterwarnings('ignore', message='.*incompatible with the GEOS version.*')

import numpy as np
import pandas as pd
import geopandas as gpd
import pygeos.creation
import rasterio
import rioxarray

from scipy.stats.mstats import gmean
from tqdm.notebook import tqdm

# enable tqdm bars for pandas.DataFrame.progress_apply
tqdm.pandas()

# Set of return periods, plus artificial lower/upper bound
RPS = np.array([1e-3, 2, 20, 50, 100, 200, 500, 1500, 1e6])

def main(event_set_path):
    # Hydrological Accumulation Zones - for river flooding
    # 'T500_ID', 'T500_Type', 'T1000_ID', 'T1000_Type', 'Country', 'Area_km2',
    # 'geometry'
    hydrological_accumulation_zones = gpd.read_file(
        "inputs/event_points_hydrological_units/JM_HAZ_T500_02.shp") \
        [['T500_ID', 'geometry']]

    # Precipitation Observation Points
    # 'op.id', 'op.lon', 'op.lat', 'region', 'sub.region', 'agg.zone'
    precipitation_ops = latlon_to_gdf(
        pd.read_csv("inputs/event_data/PrcipOPInfo.csv"),
        lat_column='op.lat', lon_column='op.lon') \
        [['op.id', 'geometry']]

    # River Observation Points (OP)
    # 'op.id', 'op.lon', 'op.lat', 'region', 'sub.region', 'agg.zone',
    # 'catchment.area', 'cent.lon', 'cent.lat'
    river_ops = latlon_to_gdf(
        pd.read_csv("inputs/event_data/RiverOPInfo.csv"),
        lat_column='op.lat', lon_column='op.lon') \
        [['op.id', 'geometry']]

    # Events - ignored
    # 'event.id', 'start.day.id', 'start.year', 'start.month', 'duration', 'extent',
    # 'is.river', 'is.precip', 'track.id', ... 'JAM'
    #
    # Note that the series of three-letter country codes columns includes JAM
    # with True/False values, but we ignore this and just include all events
    # as all seem to relate to Jamaica OPs.
    #
    # We also ignore the is.river / is.precip columns, as the *events_rp file
    # seems to disagree in terms of whether a precipitation or river OP is
    # for reporting/metadata.
    #
    # NB 'track.id' for observed events is potentially interesting as a link
    # to the IBTrACS tropical cyclone track.
    #
    # observed_events = pd.read_csv("inputs/event_data/ObsEventInfo.csv")
    # simulated_events = pd.read_csv("inputs/event_data/SimEventInfo.csv")

    # Event-OP return period
    # 'event.id', 'op.id', 'rp', 'peak.day.id', 'start.day.id', 'end.day.id'
    #
    # Future events - same set as in SimEventInfo.csv, conditioned for
    # different climate scenarios, RCP: 2.6/4.5/8.5, epoch: 2050/2080
    #
    # Files:
    # - "inputs/event_data/ObsEventRP.csv"
    # - "inputs/event_data/SimEventRP.csv"
    # - "inputs/future_event_sets/SimEventRP*.csv"
    event_set = pd.read_csv(event_set_path, usecols=['event.id', 'op.id', 'rp'])

    # Link River OPs to HAZs
    haz_to_river_within = hydrological_accumulation_zones \
        .sjoin(river_ops, predicate='contains', how='right') \
        [['T500_ID', 'op.id']]

    # some HAZ do not contain an OP
    haz_without_op = hydrological_accumulation_zones[
        ~hydrological_accumulation_zones.T500_ID.isin(haz_to_river_within.T500_ID.unique())]
    haz_to_river_nearest = haz_without_op.sjoin_nearest(river_ops, how='left') \
        [['T500_ID', 'op.id']]
    haz_to_river = pd.concat([haz_to_river_within, haz_to_river_nearest])

    river_rp_points = read_rp_maps_to_points(
        'inputs/fluvial_raw_fld_depth/JM_FLRF_UD_*-aligned.tif')
    # (T500_ID) cell_index, rp100, rp1500, rp200, rp20, rp500, rp50, rp2, geometry
    river_exposure_points = river_rp_points.sjoin(
        hydrological_accumulation_zones, predicate='within', how='left') \
        .reset_index() \
        .drop(columns='index_right') \
        .set_index('T500_ID')

    precip_rp_points = read_rp_maps_to_points(
        'inputs/surface_water_raw_fld_depth/JM_FLSW_UD_*-aligned.tif')
    # (op.id) cell_index, rp100, rp1500, rp200, rp20, rp500, rp50, rp2, geometry
    precip_exposure_points = precip_rp_points.sjoin_nearest(
        precipitation_ops, how='left') \
        .reset_index() \
        .drop(columns='index_right') \
        .set_index('op.id')

    # Set index and extract only return period column
    # (op.id, event.id) rp
    events = event_set.copy().set_index(['op.id', 'event.id'])

    # Extract precipitation events
    # (op.id, event.id) rp
    precipitation_events = events.join(precipitation_ops.set_index('op.id')) \
        .dropna() \
        .drop(columns='geometry') \
        .reset_index() \
        .set_index(['event.id', 'op.id'])

    # Calculate precipitation event exposure
    interpolate_event_exposure(
        precipitation_events, precip_exposure_points, hazard_prefix='FLSW')

    # Link event river OPs to HAZ (drop precipitation OPs which are not linked)
    # (op.id, event.id) rp, T500_ID
    river_events = events.reset_index() \
        .merge(haz_to_river, on='op.id') \
        .dropna()

    # Assert river + precip >= all (could have increased when linking river HAZ)
    assert len(river_events) + len(precipitation_events) - len(events) >= 0

    # Take the geometric mean of Event/OP return periods if multiple OPs per HAZ
    # (T500_ID, event.id) rp
    river_events_haz = river_events.groupby(['event.id', 'T500_ID']) \
        .agg({'rp': gmean})

    # Calculate river event exposure
    interpolate_event_exposure(
        river_events_haz, river_exposure_points, hazard_prefix='FLRF')


def latlon_to_gdf(df, lat_column='lat', lon_column='lon'):
    geometry = pygeos.creation.points(df[lon_column], df[lat_column])
    return gpd.GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)


def read_rp_map(fname):
    """Read flood maps, use all cells with any depth > 0 as potential exposure
    points
    """
    rp = re.search(r'Q(\d+)_', fname).group(1)
    colname = f'rp{rp}'
    with rasterio.open(fname) as dataset:
        data = dataset.read(1)
        data[data == dataset.nodata] = np.nan
        df = pd.DataFrame({colname:data.flatten()})

    df = df[df[colname] > 0].dropna()
    return df


def get_xy_df(fname):
    """Read a raster file, convert to dataframe retaining only x,y coordinate
    values  and 0..n index
    """
    xy = rioxarray.open_rasterio(fname) \
        .to_dataframe('data') \
        .reset_index() \
        .drop(columns=['band','spatial_ref', 'data'])
    return xy


def read_rp_maps_to_points(pattern):
    # Define as an iter to use each file once
    rp_maps = iter(sorted(glob(pattern)))

    # Read first
    rp_points = read_rp_map(next(rp_maps))
    # Read the rest
    for fname in rp_maps:
        df = read_rp_map(fname)
        rp_points = rp_points.join(df, how='outer')

    # Fill NA with zeros
    rp_points = rp_points.fillna(0)
    rp_points['rp2'] = 0
    xy = get_xy_df(fname)
    rp_points = rp_points.join(xy)

    # name index
    rp_points.index.rename('cell_index', inplace=True)

    # Convert to GeoDataFrame
    return latlon_to_gdf(rp_points, lat_column='y', lon_column='x') \
        .drop(columns=['y', 'x'])


def interpolate_rp_factor(df):
    return (
        (np.log(df.rp) - np.log(df.rp_l))
        / (np.log(df.rp_u) - np.log(df.rp_l)))


def interpolate_depth_df(df):
    depth = df.depth_l + (
        (df.depth_u - df.depth_l)
        * df.rp_factor)

    return depth


def interpolate_event_exposure(event_zones, exposure_points, hazard_prefix):
    # Cap at max RP 1500
    event_zones.loc[event_zones.rp >= 1500, 'rp'] = 1500

    bin_index = np.searchsorted(RPS, event_zones.rp, side='left')
    event_zones['bin_index'] = bin_index
    event_zones['rp_l'] = RPS[bin_index - 1]
    event_zones['rp_u'] = RPS[bin_index]
    event_zones['rp_factor'] = interpolate_rp_factor(event_zones)
    # event_zones is now a dataframe with:
    # (T500_ID/op.id, event.id) rp, bin_index, rp_l, rp_u, rp_factor

    for e in tqdm(event_zones.reset_index()['event.id'].unique()):
        # Each HAZ in this event, with RP values
        event_haz = event_zones.loc[e]
        # All points for this event, joined with RP values via HAZ
        event_points = exposure_points.join(event_haz).dropna()
        event_points.bin_index = event_points.bin_index.astype(np.int32)

        if len(event_points):
            depths = [
                0, event_points.rp2, event_points.rp20, event_points.rp50,
                event_points.rp100, event_points.rp200, event_points.rp500, event_points.rp1500
            ]
            event_points['depth_l'] = np.choose(event_points.bin_index - 1, depths)
            event_points['depth_u'] = np.choose(event_points.bin_index, depths)
            event_points['depth'] = interpolate_depth_df(event_points)
            # Any RP < 2 gets zero depth
            event_points.loc[event_points.rp <= 2, 'depth'] = 0

            # Output cells
            event_points = event_points[['depth', 'cell_index']]
            event_points = event_points[event_points.depth > 0]
            event_points['event'] = e
            # (T500_ID/op.id) depth, cell_index, event
            # `hazard_prefix` should include RCP, epoch metadata from event set
            event_points.to_parquet(f"outputs/{hazard_prefix}_{e}.parquet")


if __name__ == '__main__':
    try:
        event_set_path = sys.argv[1]
        print(f"Processing events defined in {event_set_path}")
    except:
        print("ERROR: did not get expected arguments")
        print("Expected usage:")
        print(f"    python {os.path.basename(__file__)} event_set.csv")

    main(event_set_path)
