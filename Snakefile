rule clip_hazard_to_HAZ:
    """Crop hazard map to HAZ bbox
    """
    input:
        tiff="processed/hazard/flrf_ud_Q{RP}.tif",
        gpkg="processed/basins/haz_500.gpkg",
    output:
        tiff="processed/event_depths/{HAZ}/hazard/flrf_ud_Q{RP}.tif"
    shell:
        """
        echo {input.tiff}
        echo {input.gpkg}
        touch {output.tiff}
        """
rule clip_exposed_asset_to_HAZ:
    """
    exposure clipped to HAZ polygon
    """
    input:
        gpkg="processed/basins/haz_500.gpkg",
        geoparquet="processed/exposure/africa_railways_network.geoparquet"
    output:
        exposed="processed/event_depths/{HAZ}/exposure/africa_railways_network.geoparquet"
    shell:
        """
        echo {input.geoparquet}
        echo {input.gpkg}
        touch {output.exposed}
        """
rule clip_exposed_asset_to_grid:
    """
    exposure clipped to HAZ polygon, and split on local (event) grid
    """
    input:
        gpkg="processed/event_depths/{HAZ}/hazard/flrf_ud_Q{RP}.tif",
        geoparquet="processed/event_depths/{HAZ}/exposure/africa_railways_network.geoparquet"
    output:
        exposed="processed/event_depths/{HAZ}/exposure/africa_railways_network_split.geoparquet"
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
        tiff="processed/event_depths/{HAZ}/hazard/flrf_ud_Q{RP}.tif",
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
rule associate_depths_to_exposure:
    """
    exposure, split on local grid, with event_depth column
    """
    input:
        tiff="processed/event_depths/{HAZ}/{EVENT}/depth.tif",
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