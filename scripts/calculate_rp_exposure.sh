SNAIL_PROGRESS=1 snail -vv process \
    --features inputs/exposure/buildings_assigned_economic_activity.csv \
    --rasters inputs/raw_fld_depth.csv

mv inputs/exposure/buildings_assigned_economic_activity.geoparquet.processed.parquet outputs/exposure/buildings__rp_depths.parquet
