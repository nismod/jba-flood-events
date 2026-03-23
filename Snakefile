import geopandas as gpd
configfile: "workflow/config.yaml"

# gdf = gpd.read_file(config["paths"]["data"] + "/basins/haz_500.gpkg")
# HAZs = gdf["T500_ID"].astype(str).tolist()
RPs = [20, 50, 100, 200, 500, 1500]
ASSET_CLASSES = ["railways", "roads", "iww", "airports", "maritime"]
asset_geoms = ["edges", "nodes", "polygons"]
# HAZ = "500_13_19495"  # For testing; replace with wildcard in rules

rule clip_to_HAZ:
    """Crop hazard map to HAZ bbox and exposed assets and defended areas clipped to HAZ polygon

    To run:
        snakemake -c1 ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_19495/defended_areas.geoparquet
    """
    input:
        script="workflow/scripts/clip_to_HAZ.py",
        # rp_tiff=expand(
        #     config["paths"]["data"] + "/hazards/flrf_ud_Q{RP}.tif",
        #     RP=RPs
        # ),
        # asset_geoparquet=expand(
        #     config["paths"]["data"] + "/infrastructure/africa_{asset}_{geom}_network.geoparquet",
        #     asset=ASSET_CLASSES,
        #     geom=asset_geoms
        # ),
        defended_areas=config["paths"]["data"] + "/defended_areas/defended_areas.gpkg",
        gpkg=config["paths"]["data"] + "/basins/haz_500.gpkg"
    output:
        # rp_clipped=expand(
        #     config["paths"]["data"] + "/event_depths/{{HAZ}}/flrf_ud_Q{RP}.tif",
        #     RP=RPs
        # ),
        # exposed_clipped=expand(
        #     config["paths"]["data"] + "/event_depths/{{HAZ}}/africa_{asset}_{geom}_network.geoparquet",
        #     asset=ASSET_CLASSES,
        #     geom=asset_geoms
        # ),
        defended_clipped=config["paths"]["data"] + "/event_depths/{{HAZ}}/defended_areas.geoparquet"
    shell:
        # echo {input.rp_tiff}
        # echo {input.asset_geoparquet}
        """
        output_path=$(dirname {output.defended_clipped})
       
        
        python {input.script} \
            --haz_id {wildcards.HAZ} \
            --haz_path {input.gpkg} \
            --defended_areas_path {input.defended_areas} \
            --output_path $(dirname {output.defended_clipped})
        """

# rule clip_all:
#     input:
#         expand(config["paths"]["data"] + "/event_depths/{HAZ}/hazard/flrf_ud_Q{RP}.tif", HAZ=HAZs, RP=RPs),
#         expand(config["paths"]["data"] + "/event_depths/{HAZ}/exposure/africa_{asset}_network.geoparquet", HAZ=HAZs, asset=ASSET_CLASSES),
#         expand(config["paths"]["data"] + "/event_depths/{HAZ}/exposure/defended_areas.geoparquet", HAZ=HAZs)

rule split_exposed_asset_to_grid:
    """
    exposure clipped to HAZ polygon, and split on local (event) grid
    """
    input:
        tiff=config["paths"]["data"] + "/event_depths/{HAZ}/hazard/flrf_ud_Q{RP}.tif",
        geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/exposure/africa_{asset}_network.geoparquet"
    output:
        exposed=config["paths"]["data"] + "/event_depths/{HAZ}/exposure/africa_{asset}_network_split.geoparquet"
    script:
        "workflow/scripts/split_asset_to_grid.py"


rule interpolate_depths:
    """
    depth_m interpolated for the RP of EVENT, in HAZ
    csv1:rp by OP by event
    csv2:lat lon of OPs
    """
    input: 
        gpkg=config["paths"]["data"] + "/basins/haz_500.gpkg",
        rp=config["paths"]["data"] + "/event_depths/{HAZ}/hazard/",  # Directory containing all RP TIFFs for this HAZ
        csv1=config["paths"]["data"] + "/events/ObsEventRp.csv",
        csv2=config["paths"]["data"] + "/events/RiverOpInfo.csv"
    output:
        depth=config["paths"]["data"] + "/event_depths/{HAZ}/{EVENT}/depth.tif"
    script:
        "workflow/scripts/depth_interpolation.py"

rule interpolate_depths_exposure_def:
    """
    depth_m interpolated for the RP of SoP, in HAZ. defended/undefended column with depth reduced according to protection standard
    exposure, split on local grid, with sop_depth column
    """
    input:
        rp20=config["paths"]["data"] + "/event_depths/{HAZ}/hazard/flrf_ud_Q20.tif",
        rp50=config["paths"]["data"] + "/event_depths/{HAZ}/hazard/flrf_ud_Q50.tif",
        rp100=config["paths"]["data"] + "/event_depths/{HAZ}/hazard/flrf_ud_Q100.tif",
        rp200=config["paths"]["data"] + "/event_depths/{HAZ}/hazard/flrf_ud_Q200.tif",
        defended=config["paths"]["data"] + "/event_depths/{HAZ}/exposure/defended_clipped.geoparquet",
        exposed=config["paths"]["data"] + "/event_depths/{HAZ}/exposure/africa_{asset}_network_split.geoparquet",
        csv1=config["paths"]["data"] + "/events/ObsEventRp.csv",
        csv2=config["paths"]["data"] + "/events/RiverOpInfo.csv"
    output:
        geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/exposure/africa_{asset}_network_split_def.geoparquet"
    shell:
        """
        echo {input.geoparquet}
        echo {input.soc_csv}
        touch {output.geoparquet}
        """

rule interpolate_depths_for_exposure:
    """
    exposure, split on local grid, with event_depth column

    output depth column is effective flood depth, reduced according to protection standard
    """
    input:
        rp20=config["paths"]["data"] + "/event_depths/{HAZ}/hazard/flrf_ud_Q20.tif",
        rp50=config["paths"]["data"] + "/event_depths/{HAZ}/hazard/flrf_ud_Q50.tif",
        rp100=config["paths"]["data"] + "/event_depths/{HAZ}/hazard/flrf_ud_Q100.tif",
        rp200=config["paths"]["data"] + "/event_depths/{HAZ}/hazard/flrf_ud_Q200.tif",
        csv1=config["paths"]["data"] + "/events/ObsEventRp.csv",
        csv2=config["paths"]["data"] + "/events/RiverOpInfo.csv",
        geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/exposure/africa_{asset}_network_def.geoparquet"
    output: 
        geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/{EVENT}/africa_{asset}_network_split_depth.geoparquet"
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
        geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/{EVENT}/africa_railways_network_split_depth.geoparquet"
    output: 
        geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/{EVENT}/africa_railways_network_split_damage.geoparquet"
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
        geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/{EVENT}/africa_railways_network_split_damage.geoparquet"
    output: 
        geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/exposure_split_with_depths.geoparquet"
    shell:
        """
        echo {input.geoparquet}
        touch {output.geoparquet}
        """