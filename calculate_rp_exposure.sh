SNAIL_PROGRESS=1 snail -vv process --features inputs/exposure/buildings_assigned_economic_activity.csv --rasters inputs/fluvial_raw_fld_depth.csv
mv inputs/exposure/buildings_assigned_economic_activity.geoparquet.processed.parquet outputs/exposure/buildings_flrf.parquet
