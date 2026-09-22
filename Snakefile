import geopandas as gpd
import pandas as pd
import os
configfile: "workflow/config.yaml"


# Constants and parameters

gdf = gpd.read_file(config["paths"]["data"] + "/basins/haz_500.gpkg")
HAZs = gdf["T500_ID"].astype(str).tolist()

TYPES = ["Obs", "Sim"] # observed or simulated events

RPs = [20, 50, 100, 200, 500, 1500]

ASSET_CLASSES = ["railway", "road", "iww", "airport", "airport_field","airport_terminal", "maritime"]

asset_geoms = ["edges", "nodes", "polygons"]
conditions = ["Undefended", "Defended"]

wildcard_constraints:
    HAZ="|".join(HAZs),
    TYPE="Obs|Sim",
    asset="|".join(ASSET_CLASSES),
    geom="edges|nodes|polygons",
    cond="Undefended|Defended"



###### clips ######
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

###### asset splits ######

rule all_splits:
    input:
        expand(
            config["paths"]["data"] + "/event_depths/{HAZ}/split_{asset}_{geom}_network.geoparquet",
            HAZ=HAZs,
            asset=ASSET_CLASSES,
            geom=asset_geoms
        )

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
###### Depths assigned to defended areas based on RP of SoP, and split to grid #######

rule split_defended_areas_to_grid:
    """
    Split defended areas to grid and assign depth based on RP of SoP
    
    
    e.g. snakemake -c1 ~/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/def/road_edges_network/split_def.geoparquet
    """
    params:
        rp_dir=config["paths"]["data"] + "/event_depths/{HAZ}/"
    input:
        script="workflow/scripts/defended_areas_to_grid.py",
        defended_geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/defended_areas.geoparquet",
        asset_geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/split_{asset}_{geom}_network.geoparquet"
    output:
        defended=config["paths"]["data"] + "/event_depths/{HAZ}/def/{asset}_{geom}_network/split_def.geoparquet"
    shell:
        """
        python {input.script} \
            --rp_path {params.rp_dir} \
            --defended_path {input.defended_geoparquet} \
            --asset_path {input.asset_geoparquet} \
            --output_path {output.defended}
        """

rule def_splits:
    """
    To run (have to be in Z drive because of file path length issues with clipped assets):
    subst Z: C://Users//cenv1075//Desktop//GitHubFiles//jba-flood-events
    cd /z/
    snakemake def_splits --cores 8 --rerun-incomplete
    """
    input:
        expand(
            config["paths"]["data"] + "/event_depths/{HAZ}/def/{asset}_{geom}_network/split_def.geoparquet",
            HAZ=HAZs,
            asset=ASSET_CLASSES,
            geom=asset_geoms
        )
###### Depth interpolation to events ######

rule interpolate_depths:
    params:
        rp_dir=config["paths"]["data"] + "/event_depths/{HAZ}/",
        output_dir=config["paths"]["data"] + "/event_depths/{HAZ}/{TYPE}Events/Undefended/{asset}_{geom}/"
    input:
        script="workflow/scripts/depth_interpolation.py",
        haz=config["paths"]["data"] + "/basins/haz_500.gpkg",
        csv=config["paths"]["data"] + "/events/{TYPE}EventRp.csv",
        gpkg=config["paths"]["data"] + "/events/RiverOpInfo.gpkg",
        geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/split_{asset}_{geom}_network.geoparquet"
    output:
        flag=config["paths"]["data"] + "/event_depths/{HAZ}/{TYPE}Events/Undefended/{asset}_{geom}/.done"
    shell:
        """
        python {input.script} \
            --haz_path {input.haz} \
            --rp_path {params.rp_dir} \
            --op_path {input.csv} \
            --info_path {input.gpkg} \
            --asset_path {input.geoparquet} \
            --output_path {params.output_dir}
        """

rule all_depths_int:
    ''''
    snakemake all_depths_int --cores 8
    '''
    input:
        expand(
            config["paths"]["data"] + "/event_depths/{HAZ}/{TYPE}Events/Undefended/{asset}_{geom}/.done",
            HAZ=HAZs,
            TYPE=TYPES,
            asset=ASSET_CLASSES,
            geom=asset_geoms
        )

###### Defended depths ######

    
rule defended_depths:
    """
    undefended depths merged with defended, undefended - sop = defended column with depth reduced according to protection standard
    snakemake -c1 "C:/Users/cenv1075/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/ObsEvents/Defended/road_edges/.done"
    """
    params:
        output_dir=config["paths"]["data"] + "/event_depths/{HAZ}/{TYPE}Events/Defended/{asset}_{geom}/",
        undefended_dir=config["paths"]["data"] + "/event_depths/{HAZ}/{TYPE}Events/Undefended/{asset}_{geom}/"
    input:
        script="workflow/scripts/defended_depths.py",
        undefended_flag=config["paths"]["data"] + "/event_depths/{HAZ}/{TYPE}Events/Undefended/{asset}_{geom}/.done",
        sop=config["paths"]["data"] + "/event_depths/{HAZ}/def/{asset}_{geom}_network/split_def.geoparquet"
    output:
        flag=config["paths"]["data"] + "/event_depths/{HAZ}/{TYPE}Events/Defended/{asset}_{geom}/.done"
    shell:
        """
        python {input.script} \
            --undefended_path {params.undefended_dir} \
            --sop_path {input.sop} \
            --output_path {params.output_dir}
        """

rule all_def_depths_int:
    ''''
    snakemake all_def_depths_int --cores 8
    '''
    input:
        expand(
            config["paths"]["data"] + "/event_depths/{HAZ}/{TYPE}Events/Defended/{asset}_{geom}/.done",
            HAZ=HAZs,
            TYPE=TYPES,
            asset=ASSET_CLASSES,
            geom=asset_geoms
        )
######


rule associate_damage_to_exposure:
    """Damage per asset element for each event, from event depths, unit costs and damage curves.
    snakemake -c1 "C:/Users/cenv1075/Desktop/DataFolders/JBA_flooding/processed_data/event_depths/500_13_33579/ObsCosts/Undefended/airport_terminal_polygons/.done"
    """
    input:
        script="workflow/scripts/associate_damage_to_exposure.py",
        depth_flag=config["paths"]["data"] + "/event_depths/{HAZ}/{TYPE}Events/{cond}/{asset}_{geom}/.done",
        asset_geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/split_{asset}_{geom}_network_prova.parquet",
        cost_table=config["paths"]["data"] + "/costs/{asset}_{geom}.csv",
    output:
        flag=touch(config["paths"]["data"] + "/event_depths/{HAZ}/{TYPE}Costs/{cond}/{asset}_{geom}/.done")
    params:
        depth_dir=lambda wildcards, input: os.path.dirname(input.depth_flag),
        out_dir=lambda wildcards, output: os.path.dirname(output.flag),
        cost_dir=config["paths"]["data"] + "/costs",
        curves_dir=config["paths"]["data"] + "/curves",
    shell:
        """
        python {input.script} \
            --depth_path {params.depth_dir} \
            --asset_path {input.asset_geoparquet} \
            --cost_path {params.cost_dir} \
            --curves_path {params.curves_dir} \
            --output_path {params.out_dir}
        """

rule all_damage_costs:
    '''
    To run (have to be in Z drive because of file path length issues with clipped assets):
    subst Z: C://Users//cenv1075//Desktop//GitHubFiles//jba-flood-events
    cd /z/
    snakemake all_damage_costs --cores 8
    '''
    input:
        expand(
            config["paths"]["data"] + "/event_depths/{HAZ}/{TYPE}Costs/{cond}/{asset}_{geom}/.done",
            HAZ=HAZs,
            TYPE=TYPES,
            cond=conditions,
            asset=ASSET_CLASSES,
            geom=asset_geoms
        )

# rule merge_damage_by_haz:
#     """
#     merge all EVENT exposure with depths into single file for HAZ:  
#         i, j, asset_id, event_1_depth, event_2_depth ...
#     """
#     input: 
#         geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/{EVENT}/africa_railways_network_split_damage.geoparquet"
#     output: 
#         geoparquet=config["paths"]["data"] + "/event_depths/{HAZ}/exposure_split_with_depths.geoparquet"
#     shell:
#         """
#         echo {input.geoparquet}
#         touch {output.geoparquet}
#         """