# rasterise zones
# note that -te and -ts are *identical* to the arguments to gdalwarp which was used to align the flood maps:
#     find inputs/**/*RD_02.tif | sed 's/.tif//' | \
#     parallel gdalwarp \
#         -t_srs EPSG:4326 \
#         -te -78.3690277774130948 17.7129166663936637 -75.9693055359931009 18.5254166728936589 \
#         -ts 8639 2925 \
#         -r near \
#         {}.tif {}-aligned.tif
gdal_rasterize \
    -a T500_INTID \
    -a_nodata -1 \
    -te -78.3690277774130948 17.7129166663936637 -75.9693055359931009 18.5254166728936589 \
    -ts 8639 2925 \
    -ot Int32 \
    -of GTiff \
    inputs/event_points_hydrological_units/JM_HAZ_T500_02.shp \
    inputs/event_points_hydrological_units/JM_HAZ_T500_02.tif

# assign flood depths and zone IDs to buildings
SNAIL_PROGRESS=1 snail -vv process \
    --features inputs/exposure/buildings_assigned_economic_activity.csv \
    --rasters inputs/raw_fld_depth.csv

mv inputs/exposure/buildings_assigned_economic_activity.geoparquet.processed.parquet outputs/exposure/buildings__rp_depths.parquet
