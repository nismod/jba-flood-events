import geopandas as gpd
gdf = gpd.read_file("processed/basins/haz_500.gpkg")
HAZ = gdf["T500_ID"].astype(str).tolist()
RP = [20, 50, 100, 200, 500, 1500]
asset = ["railways", "roads", "iww", "airports", "maritime"]

rule clip_hazard_to_HAZ:
    """Crop hazard map to HAZ bbox
    """
    input:
        tiff="processed/hazard/flrf_ud_Q{RP}.tif",
        gpkg="processed/basins/haz_500.gpkg"
    output:
        tiff="processed/event_depths/{HAZ}/hazard/flrf_ud_Q{RP}.tif"
    script:
        "scripts/workflow/intersections_HAZ_hazard.py"

rule clip_exposed_asset_to_HAZ:
    """
    exposure clipped to HAZ polygon
    """
    input:
        gpkg="processed/basins/haz_500.gpkg",
        asset_geoparquet="processed/exposure/africa_{asset}_network.geoparquet"
    output:
        exposed="processed/event_depths/{HAZ}/exposure/africa_{asset}_network.geoparquet"
    script:
        "scripts/workflow/intersections_HAZ_asset.py"

rule clip_defended_areas_to_HAZ:
    """
    exposure clipped to defended areas polygon
    """
    input:
        defended_areas="processed/defended_areas/defended.gpkg",
        haz="processed/basins/haz_500.gpkg",
    output:
        defended_areas="processed/event_depths/{HAZ}/exposure/defended_areas.geoparquet"
    
    script:
        "scripts/workflow/intersections_HAZ_defended.py"
  
rule associate_standard_of_protection:
    """
    exposure with standard_of_protection column
    """
    input:
        geoparquet="processed/event_depths/{HAZ}/exposure/africa_railways_network.geoparquet",
        defended="processed/event_depths/{HAZ}/exposure/defended_areas.geoparquet"
    output:
        geoparquet="processed/event_depths/{HAZ}/exposure/africa_railways_network_defended.geoparquet"
    shell:
        """
        echo {input.geoparquet}
        echo {input.soc_csv}
        touch {output.geoparquet}
        """
rule split_exposed_asset_to_grid:
    """
    exposure clipped to HAZ polygon, and split on local (event) grid
    """
    input:
        gpkg="processed/event_depths/{HAZ}/hazard/flrf_ud_Q{RP}.tif",
        geoparquet="processed/event_depths/{HAZ}/exposure/africa_{asset}_network_defended.geoparquet"
    output:
        exposed="processed/event_depths/{HAZ}/exposure/africa_{asset}_network_split.geoparquet"
    shell:
        """
        echo {input.geoparquet}
        echo {input.gpkg}
        touch {output.exposed}
        """
rule interpolate_depths:
    """
    depth_m interpolated for the RP of EVENT, in HAZ
    csv1:rp by OP by event
    csv2:lat lon of OPs
    """
    input: 
        rp20="processed/event_depths/{HAZ}/hazard/flrf_ud_Q20.tif",
        rp50="processed/event_depths/{HAZ}/hazard/flrf_ud_Q50.tif",
        rp100="processed/event_depths/{HAZ}/hazard/flrf_ud_Q100.tif",
        rp200="processed/event_depths/{HAZ}/hazard/flrf_ud_Q200.tif",
        csv1="processed/events/ObsEventRp.csv",
        csv2="processed/events/RiverOpInfo.csv"
    output:
        depth="processed/event_depths/{HAZ}/{EVENT}/depth.tif"

    shell:
        """
        echo {input.tiff}
        echo {input.csv1}
        echo {input.csv2}
        touch {output}
        """
rule interpolate_depths_for_exposure:
    """
    exposure, split on local grid, with event_depth column

    output depth column is effective flood depth, reduced according to protection standard
    """
    input:
        rp20="processed/event_depths/{HAZ}/hazard/flrf_ud_Q20.tif",
        rp50="processed/event_depths/{HAZ}/hazard/flrf_ud_Q50.tif",
        rp100="processed/event_depths/{HAZ}/hazard/flrf_ud_Q100.tif",
        rp200="processed/event_depths/{HAZ}/hazard/flrf_ud_Q200.tif",
        csv1="processed/events/ObsEventRp.csv",
        csv2="processed/events/RiverOpInfo.csv"
        geoparquet="processed/event_depths/{HAZ}/exposure/africa_railways_network_split.geoparquet"
    output: 
        geoparquet="processed/event_depths/{HAZ}/{EVENT}/africa_railways_network_split_depth.geoparquet"
    shell:
        """
        echo {input.tiff}
        echo {input.geoparquet}
        touch {output}
        """
rule associate_damage_to_exposure:
    """exposure with event_damage column (total of damage for each split element, 
    derived from splits, grouped by asset_id and summed)
    """
    input:
        geoparquet="processed/event_depths/{HAZ}/{EVENT}/africa_railways_network_split_depth.geoparquet"
    output: 
        geoparquet="processed/event_depths/{HAZ}/{EVENT}/africa_railways_network_split_damage.geoparquet"
    shell:
        """
        echo {input.geoparquet}
        touch {output.geoparquet}
        """
rule merge_damage_by_haz:
    """
    merge all EVENT exposure with depths into single file for HAZ:  
        i, j, asset_id, event_1_depth, event_2_depth ...
    """
    input: 
        geoparquet="processed/event_depths/{HAZ}/{EVENT}/africa_railways_network_split_damage.geoparquet"
    output: 
        geoparquet="processed/event_depths/{HAZ}/exposure_split_with_depths.geoparquet"
    shell:
        """
        echo {input.geoparquet}
        touch {output.geoparquet}
        """