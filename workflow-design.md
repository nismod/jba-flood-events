# Workflow design

## Inputs

processed/hazard/flrf_ud_Q20.tif
processed/hazard/flrf_ud_Q50.tif
processed/hazard/flrf_ud_Q100.tif
processed/hazard/flrf_ud_Q200.tif
processed/hazard/flrf_ud_Q500.tif
processed/hazard/flrf_ud_Q1500.tif

processed/regions/defended_areas.gpkg
processed/regions/regions.gpkg

processed/basins/haz_250.gpkg
processed/basins/haz_500.gpkg
processed/basins/haz_1000.gpkg

processed/events/ObsEventInfo.csv
processed/events/ObsEventRp.csv
processed/events/ObsTrack.csv
processed/events/PrecipOpInfo.csv
processed/events/RiverOpInfo.csv
processed/events/SimEventInfo.csv
processed/events/SimEventRp.csv
processed/events/SimTrack.csv

processed/exposure/africa_railways_network.geoparquet

## Outputs

processed/event_depths/{HAZ}/hazard/flrf_ud_Q{RP}.tif # hazard map cropped to HAZ bbox
processed/event_depths/{HAZ}/exposure/africa_railways_network.geoparquet # exposure clipped to HAZ polygon

processed/event_depths/{HAZ}/exposure/africa_railways_network_split.geoparquet # exposure clipped to HAZ polygon, and split on local grid

processed/event_depths/{HAZ}/{EVENT}/depth.tif # depth_m interpolated for the RP of EVENT, in HAZ
processed/event_depths/{HAZ}/{EVENT}/africa_railways_network_split.geoparquet # exposure, split on local grid, with event_depth column
processed/event_depths/{HAZ}/{EVENT}/africa_railways_network.geoparquet # exposure with event_damage column (total of damage for each split element, derived from splits, grouped by asset_id and summed)

processed/event_depths/{HAZ}/exposure_split_with_depths.geoparquet # i, j, asset_id, event_1_depth, event_2_depth ...

... later aggregation and analysis
