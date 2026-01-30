REGIONS = [R11, R12]
REGION_CODES = [20161207, 20161126] 
SCENARIOS = ["ObsEventRp", "SimEventRp"]


# rule all:
#     input:
#         "plots/quals.svg"

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

rule tif_creation

rule flood/transport overlap