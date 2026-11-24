"""Flood event exposure and damage for transport assets.

Each job reads a window of the continental hazard maps and a bbox subset of
the prepared continental asset layers. Nothing is clipped or copied per HAZ.

    hazard_grid       once            check RP maps share one grid
    prepare_assets    per layer       reproject, explode, sort, write bbox-indexed GeoParquet
    link_haz_op       once            HAZ -> OP links (within, else nearest within limit)
    haz_event_rp      per event type  (HAZ, event) -> RP, geometric mean over OPs
    split_assets      per HAZ, layer  pieces on the global grid, with mapped depths and SoP
    event_damages     per HAZ, event type, layer
                                      depth and damage per (piece, event)

Run everything:

    snakemake --cores 8 all

or on SLURM (see profiles/slurm):

    snakemake --profile profiles/slurm all
"""

import json
import os
from glob import glob

import pyogrio

configfile: "workflow/config.yaml"

# Shared data paths; override the file with `--config paths_file=...`
PATHS_FILE = config.get("paths_file", "config.json")
if os.path.exists(PATHS_FILE):
    with open(PATHS_FILE) as fh:
        config["paths"] = json.load(fh)["paths"]

DATA = config["paths"]["data"]


def data_path(*parts):
    return os.path.join(DATA, *parts)


RPS = config["hazard"]["return_periods"]
RP_PATHS = [data_path(config["hazard"]["rp_path"].format(RP=rp)) for rp in RPS]
HAZ_PATH = data_path(config["haz"]["path"])
HAZ_ID_COLUMN = config["haz"]["id_column"]
TYPES = config["events"]["types"]
DEFENDED_PATH = data_path(config["defended"]["path"]) if config["defended"]["path"] else None

HAZS = sorted(
    pyogrio.read_dataframe(HAZ_PATH, columns=[HAZ_ID_COLUMN], read_geometry=False)[HAZ_ID_COLUMN]
    .astype(str)
    .unique()
)

if config["assets"]["layers"]:
    LAYERS = list(config["assets"]["layers"])
else:
    LAYERS = sorted(
        layer
        for layer in glob_wildcards(data_path(config["assets"]["source_path"])).layer
        if layer.rsplit("_", 1)[-1] in ("edges", "nodes", "polygons")
    )

wildcard_constraints:
    HAZ=r"[^/]+",
    TYPE="|".join(TYPES),
    layer="|".join(LAYERS) if LAYERS else r"[^/]+",


def asset_type(layer):
    return layer.rsplit("_", 1)[0]


def cost_table(layer):
    return data_path(config["damage"]["cost_path"].format(layer=layer))


def curves_dir(layer):
    return data_path(config["damage"]["curves_dir"].format(asset_type=asset_type(layer)))


def rp_args():
    return " ".join(f"--rp {rp} --rp_path {path}" for rp, path in zip(RPS, RP_PATHS))


rule all:
    input:
        expand(
            data_path("event_damages", "{TYPE}", "{layer}", "haz={HAZ}", "part.parquet"),
            TYPE=TYPES,
            layer=LAYERS,
            HAZ=HAZS,
        ),


rule all_splits:
    input:
        expand(data_path("exposure", "{layer}", "haz={HAZ}", "splits.parquet"), layer=LAYERS, HAZ=HAZS),


rule hazard_grid:
    input:
        script="workflow/scripts/hazard_grid.py",
        rps=RP_PATHS,
    output:
        data_path("prepared", "hazard_grid.json"),
    shell:
        "python {input.script} --output {output} {input.rps}"


rule prepare_assets:
    input:
        script="workflow/scripts/prepare_assets.py",
        assets=data_path(config["assets"]["source_path"]),
        grid=rules.hazard_grid.output,
    output:
        data_path("prepared", "assets", "{layer}.parquet"),
    params:
        id_column=config["assets"]["id_column"],
    resources:
        mem_mb=32000,
    shell:
        """
        python {input.script} --layer {wildcards.layer} --asset_path {input.assets} \
            --grid {input.grid} --id_column {params.id_column} --output {output}
        """


rule link_haz_op:
    input:
        script="workflow/scripts/link_haz_op.py",
        haz=HAZ_PATH,
        ops=data_path(config["events"]["op_path"]),
    output:
        data_path("prepared", "events", "haz_op_links.parquet"),
    params:
        max_distance_m=config["events"]["nearest_op_max_distance_m"],
        distance_crs=config["events"]["distance_crs"],
    shell:
        """
        python {input.script} --haz_path {input.haz} --haz_id_column {HAZ_ID_COLUMN} \
            --op_path {input.ops} --max_distance_m {params.max_distance_m} \
            --distance_crs '{params.distance_crs}' --output {output}
        """


rule haz_event_rp:
    input:
        script="workflow/scripts/haz_event_rp.py",
        events=data_path(config["events"]["rp_path"]),
        links=rules.link_haz_op.output,
    output:
        data_path("prepared", "events", "{TYPE}_haz_event_rp.parquet"),
    resources:
        mem_mb=16000,
    shell:
        "python {input.script} --events_path {input.events} --links_path {input.links} --output {output}"


rule split_assets:
    input:
        script="workflow/scripts/split_assets.py",
        haz=HAZ_PATH,
        grid=rules.hazard_grid.output,
        rps=RP_PATHS,
        assets=rules.prepare_assets.output,
        defended=[DEFENDED_PATH] if DEFENDED_PATH else [],
    output:
        data_path("exposure", "{layer}", "haz={HAZ}", "splits.parquet"),
    params:
        rp_args=rp_args(),
        defended=f"--defended_path {DEFENDED_PATH} --sop_column {config['defended']['sop_column']}"
        if DEFENDED_PATH
        else "",
    resources:
        mem_mb=config["resources"]["split_assets_mem_mb"],
    shell:
        """
        python {input.script} --haz_id {wildcards.HAZ} --haz_path {input.haz} \
            --haz_id_column {HAZ_ID_COLUMN} --grid {input.grid} {params.rp_args} \
            --layer {wildcards.layer} --asset_path {input.assets} {params.defended} \
            --output {output}
        """


rule event_damages:
    input:
        script="workflow/scripts/event_damages.py",
        splits=rules.split_assets.output,
        haz_event_rp=rules.haz_event_rp.output,
        # optional: a layer with no cost table must be exposure-only (no cost_type)
        cost_table=lambda wc: [cost_table(wc.layer)] if os.path.exists(cost_table(wc.layer)) else [],
        curves=lambda wc: sorted(glob(os.path.join(curves_dir(wc.layer), "*.csv"))),
    output:
        data_path("event_damages", "{TYPE}", "{layer}", "haz={HAZ}", "part.parquet"),
    params:
        rp_args=" ".join(f"--rp {rp}" for rp in RPS),
        horp=config["hazard"]["horp"],
        cost_table=lambda wc: cost_table(wc.layer),
        curves_dir=lambda wc: curves_dir(wc.layer),
    resources:
        mem_mb=config["resources"]["event_damages_mem_mb"],
    shell:
        """
        python {input.script} --haz_id {wildcards.HAZ} --splits {input.splits} \
            --haz_event_rp {input.haz_event_rp} {params.rp_args} --horp {params.horp} \
            --layer {wildcards.layer} --cost_table {params.cost_table} \
            --curves_dir {params.curves_dir} --output {output}
        """
