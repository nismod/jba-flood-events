"""Flood event exposure, damage and risk for transport assets.

Each per-HAZ job reads a window of the continental hazard maps and a bbox
subset of the prepared continental asset layers. Nothing is clipped or copied
per HAZ. See workflow-design.md.

    validate_inputs      once              check inputs, fail early with every problem found
    hazard_grid          once              check RP maps share one grid
    prepare_assets       per layer         reproject, explode, sort, write bbox-indexed GeoParquet
    link_haz_op          once              HAZ -> OP links (within, else nearest within limit)
    haz_event_rp         per event set     (HAZ, event) -> RP, geometric mean over OPs
    split_assets         per HAZ           pieces of every layer on the global grid
    event_damages        per HAZ           depth and damage per (asset, adm1, event), all layers and event sets
    aggregate_losses     per set, layer    event losses by HAZ, adm1, adm0
    risk_metrics         per layer + all   expected annual and exceedance values
    qa_summary           once              exposure per layer and HAZ
    disconnection_*      draft             population disconnected per event, and its risk

Run everything:

    snakemake --profile profiles/local all     # one machine
    snakemake --profile profiles/slurm all     # SLURM
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


def optional_path(relative):
    return data_path(relative) if relative else None


RPS = config["hazard"]["return_periods"]
RP_PATHS = [data_path(config["hazard"]["rp_path"].format(RP=rp)) for rp in RPS]
HAZ_PATH = data_path(config["haz"]["path"])
HAZ_ID_COLUMN = config["haz"]["id_column"]
TYPES = config["events"]["types"]
DEFENDED_PATH = optional_path(config["defended"]["path"])
ADMIN_PATH = optional_path(config["admin"]["path"])
RISK_TYPE = config["risk"]["event_type"]
DISCONNECTION = config["disconnection"]

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
    chunk=r"\d+",

# Path templates. {{layer}} and {{TYPE}} survive as {layer}, {TYPE} for the scripts.
SOURCE_ASSETS = data_path(config["assets"]["source_path"])
PREPARED_ASSETS = data_path("prepared", "assets", "{layer}.parquet")
SPLITS = data_path("exposure", "{layer}", "haz={HAZ}", "splits.parquet")
HAZ_EVENT_RP = data_path("prepared", "events", "{TYPE}_haz_event_rp.parquet")
ASSET_EVENTS = data_path("asset_event_damages", "{TYPE}", "{layer}", "haz={HAZ}", "part.parquet")
EVENT_LOSSES = data_path("event_losses", "{TYPE}", "{layer}")
COST_TABLE = data_path(config["damage"]["cost_path"])
CURVES_DIR = data_path(config["damage"]["curves_dir"])
LEVELS = ["haz", "adm1", "adm0"]
RISK_FILES = ["{level}_expected_annual.parquet", "{level}_exceedance.parquet"]


def asset_type(layer):
    return layer.rsplit("_", 1)[0]


def flags(name, values):
    return " ".join(f"--{name} {value}" for value in values)


def admin_flags(adm0=True):
    if not ADMIN_PATH:
        return ""
    out = f"--admin_path {ADMIN_PATH} --adm1_column {config['admin']['adm1_column']}"
    return out + (f" --adm0_column {config['admin']['adm0_column']}" if adm0 else "")


DEFENDED_FLAGS = (
    f"--defended_path {DEFENDED_PATH} --sop_column {config['defended']['sop_column']}" if DEFENDED_PATH else ""
)
PROBABILITY_FLAGS = flags("probability", config["risk"]["exceedance_probabilities"])


def risk_outputs(directory, levels):
    return expand(os.path.join(directory, "{file}"), file=expand(RISK_FILES, level=levels), allow_missing=True)


def disconnection_targets():
    if not DISCONNECTION["enabled"]:
        return []
    return risk_outputs(data_path("disconnection", RISK_TYPE, "risk"), ["adm1", "adm0", "total"])


rule all:
    input:
        expand(ASSET_EVENTS, TYPE=TYPES, layer=LAYERS, HAZ=HAZS),
        expand(os.path.join(EVENT_LOSSES, "{level}.parquet"), TYPE=TYPES, layer=LAYERS, level=LEVELS),
        [risk_outputs(data_path("risk", layer), ["asset", *LEVELS]) for layer in LAYERS],
        risk_outputs(data_path("risk", "all"), LEVELS),
        data_path("qa", "exposure_summary.csv"),
        disconnection_targets(),


rule all_splits:
    input:
        expand(SPLITS, layer=LAYERS, HAZ=HAZS),


rule validate_inputs:
    input:
        script="workflow/scripts/validate_inputs.py",
        haz=HAZ_PATH,
        ops=data_path(config["events"]["op_path"]),
        events=expand(data_path(config["events"]["rp_path"]), TYPE=TYPES),
        assets=expand(SOURCE_ASSETS, layer=LAYERS),
        costs=[p for p in expand(COST_TABLE, layer=LAYERS) if os.path.exists(p)],
        curves=[f for layer in LAYERS for f in sorted(glob(os.path.join(CURVES_DIR.format(asset_type=asset_type(layer)), "*.csv")))],
        defended=[DEFENDED_PATH] if DEFENDED_PATH else [],
        admin=[ADMIN_PATH] if ADMIN_PATH else [],
    output:
        data_path("prepared", "validation.json"),
    params:
        events=lambda wc, input: flags("events_path", input.events),
        layers=flags("layer", LAYERS),
        admin=admin_flags(),
    resources:
        mem_mb=16000,
    shell:
        """
        python {input.script} --haz_path {input.haz} --haz_id_column {HAZ_ID_COLUMN} \
            --op_path {input.ops} {params.events} {params.layers} \
            --asset_template '{SOURCE_ASSETS}' --id_column {config[assets][id_column]} \
            --cost_template '{COST_TABLE}' --curves_template '{CURVES_DIR}' \
            {DEFENDED_FLAGS} {params.admin} --output {output}
        """


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
        assets=SOURCE_ASSETS,
        grid=rules.hazard_grid.output,
        valid=rules.validate_inputs.output,
    output:
        PREPARED_ASSETS,
    resources:
        mem_mb=32000,
    shell:
        """
        python {input.script} --layer {wildcards.layer} --asset_path {input.assets} \
            --grid {input.grid} --id_column {config[assets][id_column]} --output {output}
        """


rule link_haz_op:
    input:
        script="workflow/scripts/link_haz_op.py",
        haz=HAZ_PATH,
        ops=data_path(config["events"]["op_path"]),
        valid=rules.validate_inputs.output,
    output:
        data_path("prepared", "events", "haz_op_links.parquet"),
    shell:
        """
        python {input.script} --haz_path {input.haz} --haz_id_column {HAZ_ID_COLUMN} \
            --op_path {input.ops} --max_distance_m {config[events][nearest_op_max_distance_m]} \
            --distance_crs '{config[events][distance_crs]}' --output {output}
        """


rule haz_event_rp:
    input:
        script="workflow/scripts/haz_event_rp.py",
        events=data_path(config["events"]["rp_path"]),
        links=rules.link_haz_op.output,
    output:
        HAZ_EVENT_RP,
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
        assets=expand(PREPARED_ASSETS, layer=LAYERS),
        defended=[DEFENDED_PATH] if DEFENDED_PATH else [],
        admin=[ADMIN_PATH] if ADMIN_PATH else [],
    output:
        expand(SPLITS, layer=LAYERS, allow_missing=True),
    params:
        rps=" ".join(f"--rp {rp} --rp_path {path}" for rp, path in zip(RPS, RP_PATHS)),
        layers=flags("layer", LAYERS),
        admin=admin_flags(adm0=False),
        output_template=lambda wc: SPLITS.replace("{HAZ}", wc.HAZ),
    resources:
        mem_mb=config["resources"]["split_assets_mem_mb"],
    shell:
        """
        python {input.script} --haz_id {wildcards.HAZ} --haz_path {input.haz} \
            --haz_id_column {HAZ_ID_COLUMN} --grid {input.grid} {params.rps} {params.layers} \
            --asset_template '{PREPARED_ASSETS}' {DEFENDED_FLAGS} {params.admin} \
            --output_template '{params.output_template}'
        """


rule event_damages:
    input:
        script="workflow/scripts/event_damages.py",
        splits=expand(SPLITS, layer=LAYERS, allow_missing=True),
        haz_event_rp=expand(HAZ_EVENT_RP, TYPE=TYPES),
        costs=rules.validate_inputs.input.costs,
        curves=rules.validate_inputs.input.curves,
    output:
        expand(ASSET_EVENTS, TYPE=TYPES, layer=LAYERS, allow_missing=True),
    params:
        layers=flags("layer", LAYERS),
        types=flags("type", TYPES),
        rps=flags("rp", RPS),
        splits_template=lambda wc: SPLITS.replace("{HAZ}", wc.HAZ),
        output_template=lambda wc: ASSET_EVENTS.replace("{HAZ}", wc.HAZ),
    resources:
        mem_mb=config["resources"]["event_damages_mem_mb"],
    shell:
        """
        python {input.script} --haz_id {wildcards.HAZ} {params.layers} {params.types} \
            --splits_template '{params.splits_template}' --haz_event_rp_template '{HAZ_EVENT_RP}' \
            {params.rps} --horp {config[hazard][horp]} --cost_template '{COST_TABLE}' \
            --curves_template '{CURVES_DIR}' --output_template '{params.output_template}'
        """


rule aggregate_losses:
    input:
        script="workflow/scripts/aggregate_losses.py",
        parts=expand(ASSET_EVENTS, HAZ=HAZS, allow_missing=True),
        admin=[ADMIN_PATH] if ADMIN_PATH else [],
    output:
        expand(os.path.join(EVENT_LOSSES, "{level}.parquet"), level=LEVELS, allow_missing=True),
    params:
        asset_events=lambda wc: data_path("asset_event_damages", wc.TYPE, wc.layer),
        admin=admin_flags(),
        output_dir=EVENT_LOSSES,
    resources:
        mem_mb=config["resources"]["risk_mem_mb"],
    shell:
        """
        python {input.script} --asset_events {params.asset_events} {params.admin} \
            --output_dir {params.output_dir}
        """


rule risk_metrics:
    input:
        script="workflow/scripts/risk_metrics.py",
        parts=expand(ASSET_EVENTS, TYPE=RISK_TYPE, HAZ=HAZS, allow_missing=True),
        losses=expand(os.path.join(EVENT_LOSSES, "{level}.parquet"), TYPE=RISK_TYPE, level=LEVELS, allow_missing=True),
    output:
        risk_outputs(data_path("risk", "{layer}"), ["asset", *LEVELS]),
    params:
        asset_events=lambda wc: data_path("asset_event_damages", RISK_TYPE, wc.layer),
        losses_dir=lambda wc: EVENT_LOSSES.format(TYPE=RISK_TYPE, layer=wc.layer),
        output_dir=lambda wc: data_path("risk", wc.layer),
    resources:
        mem_mb=config["resources"]["risk_mem_mb"],
    shell:
        """
        python {input.script} --years {config[risk][years]} {PROBABILITY_FLAGS} \
            --value damage_undefended_usd --value damage_defended_usd \
            --value exposed_size --value exposed_size_defended \
            --asset_events {params.asset_events} --event_losses_dir {params.losses_dir} \
            --output_dir {params.output_dir}
        """


rule risk_metrics_all_layers:
    input:
        script="workflow/scripts/risk_metrics.py",
        losses=expand(os.path.join(EVENT_LOSSES, "{level}.parquet"), TYPE=RISK_TYPE, layer=LAYERS, level=LEVELS),
    output:
        risk_outputs(data_path("risk", "all"), LEVELS),
    params:
        losses_dirs=flags("event_losses_dir", expand(EVENT_LOSSES, TYPE=RISK_TYPE, layer=LAYERS)),
        output_dir=data_path("risk", "all"),
    shell:
        """
        python {input.script} --years {config[risk][years]} {PROBABILITY_FLAGS} \
            --value damage_undefended_usd --value damage_defended_usd \
            {params.losses_dirs} --output_dir {params.output_dir}
        """


rule qa_summary:
    input:
        script="workflow/scripts/qa_summary.py",
        splits=expand(SPLITS, layer=LAYERS, HAZ=HAZS),
    output:
        data_path("qa", "exposure_summary.csv"),
    params:
        rps=flags("rp", RPS),
        layer_dirs=" ".join(f"{layer}={data_path('exposure', layer)}" for layer in LAYERS),
    shell:
        "python {input.script} {params.rps} --output {output} {params.layer_dirs}"


# Draft population-disconnected metric

DISCONNECTION_BASELINE = data_path("disconnection", "baseline")
DISCONNECTION_EVENTS = data_path("disconnection", "{TYPE}", "events", "part-{chunk}.parquet")
NETWORK_LAYER = DISCONNECTION["network_layer"]


rule disconnection_baseline:
    input:
        script="workflow/scripts/disconnection_baseline.py",
        network=PREPARED_ASSETS.format(layer=NETWORK_LAYER),
        population=data_path(DISCONNECTION["population_path"]),
        centres=data_path(DISCONNECTION["centres_path"]),
        admin=[ADMIN_PATH] if ADMIN_PATH else [],
    output:
        expand(
            os.path.join(DISCONNECTION_BASELINE, "{file}"),
            file=["edges.parquet", "nodes.parquet", "sources.parquet", "cells.parquet", "population.parquet", "summary.json"],
        ),
    params:
        admin=admin_flags(),
    resources:
        mem_mb=config["resources"]["disconnection_mem_mb"],
    shell:
        """
        python {input.script} --network {input.network} --population {input.population} \
            --centres {input.centres} --centre_id_column {DISCONNECTION[centre_id_column]} \
            {params.admin} --max_snap_m {DISCONNECTION[max_snap_m]} --output_dir {DISCONNECTION_BASELINE}
        """


rule disconnection_events:
    input:
        script="workflow/scripts/disconnection_events.py",
        baseline=rules.disconnection_baseline.output,
        parts=expand(ASSET_EVENTS, layer=NETWORK_LAYER, HAZ=HAZS, allow_missing=True),
    output:
        DISCONNECTION_EVENTS,
    params:
        asset_events=lambda wc: data_path("asset_event_damages", wc.TYPE, NETWORK_LAYER),
    resources:
        mem_mb=config["resources"]["disconnection_mem_mb"],
    shell:
        """
        python {input.script} --baseline_dir {DISCONNECTION_BASELINE} --asset_events {params.asset_events} \
            --threshold_m {DISCONNECTION[depth_threshold_m]} --chunk {wildcards.chunk} \
            --n_chunks {DISCONNECTION[chunks]} --output {output}
        """


rule disconnection_risk:
    input:
        script="workflow/scripts/disconnection_risk.py",
        events=expand(DISCONNECTION_EVENTS, TYPE=RISK_TYPE, chunk=range(DISCONNECTION["chunks"])),
    output:
        risk_outputs(data_path("disconnection", RISK_TYPE, "risk"), ["adm1", "adm0", "total"]),
    params:
        output_dir=data_path("disconnection", RISK_TYPE, "risk"),
    shell:
        """
        python {input.script} --years {config[risk][years]} {PROBABILITY_FLAGS} \
            --output_dir {params.output_dir} {input.events}
        """
