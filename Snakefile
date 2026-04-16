import geopandas as gpd
configfile: "workflow/config.yaml"

gdf = gpd.read_file(config["paths"]["data"] + "/basins/haz_500.gpkg")
HAZs = gdf["T500_ID"].astype(str).tolist()

ev_id = pd.read_csv(config["paths"]["data"] + "/events/RiverOpInfo.csv")
EVENTS = ev_id["event.id"].astype(str).unique().tolist()

TYPES = ["Obs", "Sim"] # observed or simulated events

RPs = [20, 50, 100, 200, 500, 1500]

ASSET_CLASSES = ["railway", "road", "iww", "airport", "airport_field","airport_terminal", "maritime"]

asset_geoms = ["edges", "nodes", "polygons"]
# HAZ = "500_13_19495"  
wildcard_constraints:
    asset="|".join(ASSET_CLASSES)

        
rule clip_all:  
    """
    To run:
        snakemake clip_all --cores 4 
    """
    input:
        expand(config["paths"]["data"] + "/event_depths/{HAZ}/defended_areas.geoparquet", HAZ=HAZs),
        expand(config["paths"]["data"] + "/event_depths/{HAZ}/flrf_ud_Q{RP}.tif", HAZ=HAZs, RP=RPs),
        expand(config["paths"]["data"] + "/event_depths/{HAZ}/{asset}_{geom}_network.geoparquet", HAZ=HAZs, asset=ASSET_CLASSES, geom=asset_geoms)
        
rule clip_def_to_HAZ:
    """Defended areas clipped to HAZ polygon

    To run:
        snakemake -c1 C:/Users/cenv1075/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_19495/defended_areas.geoparquet
    """
    input:
        script="workflow/scripts/clip_def_to_HAZ.py",
        defended_areas=config["paths"]["data"] + "/defended_areas/defended_areas.gpkg",
        gpkg=config["paths"]["data"] + "/basins/haz_500.gpkg"
    output:
        defended_clipped=config["paths"]["data"] + "/event_depths/{HAZ}/defended_areas.geoparquet"
    shell:
        # echo {input.rp_tiff}
        # echo {input.asset_geoparquet}
        # output_path=$(dirname {output.defended_clipped})
        """
        python {input.script} \
            --haz_id {wildcards.HAZ} \
            --haz_path {input.gpkg} \
            --defended_areas_path {input.defended_areas} \
            --output_path {output.defended_clipped}
        """
rule clip_rp_to_HAZ:
    """Crop hazard map to HAZ bbox

    To run:
        snakemake -c1 ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_19495/flrf_ud_Q500.tif
    """
    input:
        script="workflow/scripts/clip_rp_to_HAZ.py",
        rp_tiff=config["paths"]["data"] + "/hazards/flrf_ud_Q{RP}.tif",
        gpkg=config["paths"]["data"] + "/basins/haz_500.gpkg"
    output:
        rp_clipped=config["paths"]["data"] + "/event_depths/{HAZ}/flrf_ud_Q{RP}.tif"
    shell:
        # echo {input.rp_tiff}
        # echo {input.asset_geoparquet}
        """      
        
        python {input.script} \
            --haz_id {wildcards.HAZ} \
            --haz_path {input.gpkg} \
            --rp_path {input.rp_tiff} \
            --output_path {output.rp_clipped}
        """

rule clip_asset_to_HAZ:
    """Crop assets to HAZ polygon

    To run:
        snakemake -c1 ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_19495/road_edges_network.parquet
    """
    input:
        script="workflow/scripts/clip_asset_to_HAZ.py",
        # Removed asset_geoparquet from input to avoid Snakemake's existence check
        gpkg=config["paths"]["data"] + "/basins/haz_500.gpkg"
    params:
        asset_path=lambda wildcards: config["paths"]["data"] + f"/infrastructure/africa_{wildcards.asset}_{wildcards.geom}_network.parquet"
    output:
        exposed_clipped=config["paths"]["data"] + "/event_depths/{HAZ}/{asset}_{geom}_network.geoparquet"
    shell:
        # echo {input.rp_tiff}
        # echo {input.asset_geoparquet}
        """
        python {input.script} \
            --haz_id {wildcards.HAZ} \
            --haz_path {input.gpkg} \
            --asset_path {params.asset_path} \
            --output_path {output.exposed_clipped}
        """

rule split_exposed_asset_to_grid:
    """
    Split exposed assets to grid

    To run (have to be in Z drive because of file path length issues with clipped assets):
    subst Z: "jba_flood_events path, double slashes"
    cd /z/
    snakemake all_splits --cores 8 --rerun-incomplete
    """
    input:
        script="workflow/scripts/exposed_assets_to_grid.py",
        tiff=config["paths"]["data"] + "/event_depths/{HAZ}/flrf_ud_Q1500.tif",
        asset_geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/{asset}_{geom}_network.geoparquet"
    output:
        exposed=config["paths"]["data"] + "/event_depths/{HAZ}/split_{asset}_{geom}_network.geoparquet"
    shell:
        """
        python {input.script} \
            --rp_path {input.tiff} \
            --asset_path {input.asset_geoparquet} \
            --output_path {output.exposed}
        """
rule all_splits:
    input:
        expand(
            config["paths"]["data"] + "/event_depths/{HAZ}/split_{asset}_{geom}_network.geoparquet",
            HAZ=HAZs,
            asset=ASSET_CLASSES,
            geom=asset_geoms
        )



checkpoint interpolate_depths:
    input:
        script="workflow/scripts/depth_interpolation.py",
        haz=config["paths"]["data"] + "/basins/haz_500.gpkg",
        rp_folder=config["paths"]["data"] + "/event_depths/{HAZ}/",
        csv=config["paths"]["data"] + "/events/{TYPE}EventRp.csv",
        gpkg=config["paths"]["data"] + "/events/RiverOpInfo.gpkg"
    output:
        outdir=directory(config["paths"]["data"] + "/event_depths/{HAZ}/{TYPE}Events/")
    shell:
        """
        python {input.script} \
            --haz_path {input.haz} \
            --rp_path {input.rp_folder} \
            --op_path {input.csv} \
            --info_path {input.gpkg} \
            --output_path {output.outdir}
        """

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