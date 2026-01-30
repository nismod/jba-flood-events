
rule countries:
    input:
        "incoming_data/country_codes.xlsx"
    output:
        "processed_data/{country}_paths.xlsx"
    script:
        "scripts/paths_gen.py"

rule flood_gen:
    input:
        "processed_data/paths.xlsx"
    output:
        "processed_data/outputs/ObsEventRp_20161207_{country}_river.csv"
        "processed_data/outputs/{country}/{country}_HAZ_depths.geoparquet"
    script:
        "scripts/flood_event_generation_AFR.py"

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
