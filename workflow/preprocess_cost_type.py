"""
Adds a `cost_type` column to infrastructure asset files (roads, railways,
airports, maritime ports, iww) based on the mapping rules in read_me.txt.

"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import os
import json

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = os.path.join(script_dir,'..', 'config.json')

    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
    return config

# ------------------------------------------------------------------
# ROADS - edges
# ------------------------------------------------------------------

def assign_road_cost_type(gdf,bridge_col,paved_col,):
    """
    gravel_good : bridge = False, paved = False
    sbst_good   : bridge = False, paved = True
    bridge      : bridge = True
    """
    

    is_bridge = gdf[bridge_col] == True  
    is_paved = gdf[paved_col] == True  

    conditions = [
        is_bridge,
        (~is_bridge) & is_paved,
        (~is_bridge) & (~is_paved),
    ]

    choices = ["bridge", "roads_paved", "roads_unpaved"]
    gdf["cost_type"] = np.select(conditions, choices, default="")
    gdf["cost_type"] = gdf["cost_type"].replace("", np.nan)

    return gdf


# ------------------------------------------------------------------
# RAILWAYS - edges
# ------------------------------------------------------------------

def assign_railway_edge_cost_type(gdf, gauge_col, structure_col, status_col, mgr_threshold):
    """
    mgr_track            : gauge <= 1067, structure is not {bridge, viaduct}
    mgr_bridge            : gauge <= 1067, structure in {bridge, viaduct}
    sgr_track_{status}    : all other gauge values (incl. NULL), structure is NULL,
                             status in {construction, open, planned, proposed, rehabilitation}
                             (rehabilitation is relabeled as construction)
    bridge                : all other gauge values (incl. NULL),
                             structure in {bridge, viaduct},
                             status = open
    """
    gauge = pd.to_numeric(gdf[gauge_col], errors="coerce")

    # ASSUMPTION: NULL/NaN gauge falls under "all other gauge values" (sgr bucket)
    is_mgr = gauge <= mgr_threshold
    is_sgr = ~is_mgr

    structure = gdf[structure_col]
    
    is_bridge_structure = structure.isin(["bridge", "viaduct"])
    is_track = ~is_bridge_structure

    status = gdf[status_col]
    valid_sgr_status = status.isin(["construction", "open", "planned", "proposed", "rehabilitation"])
    is_open = status == "open"

    # ASSUMPTION: rehabilitation status is costed the same as construction
    status_for_label = status.replace("rehabilitation", "construction")
    sgr_track_label = "sgr_track_" + status_for_label.astype(str)

    conditions = [
        is_mgr & is_track,
        is_mgr & is_bridge_structure,
        is_sgr & is_track & valid_sgr_status,
        is_sgr & is_bridge_structure & is_open, # not included planned/proposed/etc. sgr track already overstimates the costs 
    ]
    choices = [
        "mgr_track",
        "mgr_bridge",
        sgr_track_label,
        "mgr_bridge",
    ]

    gdf["cost_type"] = np.select(conditions, choices, default="")
    gdf["cost_type"] = gdf["cost_type"].replace("", np.nan)
    return gdf


# ------------------------------------------------------------------
# RAILWAYS - nodes (exposure-only, no costing)
# ------------------------------------------------------------------

def assign_other_infra_cost_type(gdf):
    """
    all nodes will have no cost_type,
    left as NaN so downstream costing code skips these rows. Railway nodes (various stations) will be reported in an appendix for exposure, but not used for costing.
    """
    gdf["cost_type"] = np.nan
    return gdf


# ------------------------------------------------------------------
# AIRPORTS
# ------------------------------------------------------------------

def assign_airport_terminal_cost_type(gdf):
    """airport (terminal polygons) -> airport_terminal"""
    gdf["cost_type"] = "airport_terminal"
    return gdf


def assign_airport_field_cost_type(gdf):
    """airport fields (runway/apron polygons) -> roads_unpaved"""
    gdf["cost_type"] = "roads_unpaved"
    return gdf

# ------------------------------------------------------------------
# IWW 
# ------------------------------------------------------------------

def assign_iww_cost_type(gdf):
    """IWW polygons -> general cargo"""
    gdf["cost_type"] = "general cargo"
    return gdf

# ------------------------------------------------------------------
# MARITIME PORTS
# ------------------------------------------------------------------


def assign_maritime_port_cost_type(gdf,category_col,):
    """
    Maps a port category column to cost_type per PORT_CATEGORY_MAP.
    Missing/unmapped values -> "general cargo" (read_me: "no value = general cargo").
    """
    normalized = gdf[category_col].astype(str).str.strip().str.lower()
    gdf["cost_type"] = normalized.map(PORT_CATEGORY_MAP)
    gdf["cost_type"] = gdf["cost_type"].fillna("general cargo")
    return gdf


# ------------------------------------------------------------------
# DISPATCH
# ------------------------------------------------------------------

# Maps (asset, subtype) parsed from filename -> assignment function.
# subtype is one of: nodes, edges, polygons (or ports/terminal/field
# for the asset types that need a finer split than nodes/edges/polygons).
DISPATCH = {
    ("road", "edges"): assign_road_cost_type,
    ("railway", "edges"): assign_railway_edge_cost_type,
    ("airport", "terminal"): assign_airport_terminal_cost_type,
    ("airport", "field"): assign_airport_field_cost_type,
    ("iww", "polygons"): assign_iww_cost_type,
    ("maritime", "polygons"): assign_maritime_port_cost_type,
    # exposure-only, no costing
    ("road", "nodes"): assign_other_infra_cost_type,
    ("railway", "nodes"): assign_other_infra_cost_type,
    ("airport", "nodes"): assign_other_infra_cost_type,
    ("iww", "nodes"): assign_other_infra_cost_type,
    ("maritime", "nodes"): assign_other_infra_cost_type,
}




def parse_asset_subtype(filename):
    """
    asset is always parts[0]; the last remaining token is the geometry
    type (nodes/edges/polygons). Anything between asset and geometry
    (e.g. "field", "terminal") is a finer-grained subtype qualifier —
    except for nodes files, which are always exposure-only regardless
    of any qualifier (e.g. airport_terminal_nodes -> ("airport","nodes")).
    """
    stem = Path(filename).stem
    parts = [p for p in stem.split("_") if p != "network"]
    if parts and parts[0] == "africa":  # extend with other region prefixes as needed
        parts = parts[1:]

    if len(parts) < 2:
        raise ValueError(
            f"Could not determine (asset, subtype) from filename '{filename}'. "
            f"Parsed tokens: {parts}."
        )

    asset = parts[0]
    rest = parts[1:]

    if rest[-1] not in ("nodes", "edges", "polygons"):
        raise ValueError(
            f"Could not determine geometry type from filename '{filename}'. "
            f"Parsed tokens: {parts}."
        )
    geom = rest[-1]
    middle = rest[:-1]

    if geom == "nodes":
        return asset, "nodes"

    subtype = middle[0] if middle else geom
    return asset, subtype

def process_file(input_path, output_path) -> gpd.GeoDataFrame:
    """Read a parquet asset file, assign cost_type, write it back out."""
    input_path = Path(input_path)
    asset, subtype = parse_asset_subtype(input_path.name)

    func = DISPATCH.get((asset, subtype))
    if func is None:
        raise ValueError(f"No handler registered for (asset='{asset}', subtype='{subtype}')")

    gdf = gpd.read_parquet(input_path)
    gdf = func(gdf)

    if output_path is None:
        output_path = input_path  # overwrite the original file

    gdf.to_parquet(output_path)
    print(f"[{asset}/{subtype}] {input_path.name} -> {output_path} "
          f"({len(gdf)} rows, {gdf['cost_type'].isna().sum()} rows with no cost_type)")
    return gdf



def process_directory(input_dir):
    """Process every *_network.parquet file in a directory (overwrites in place)."""
    input_dir = Path(input_dir)

    for f in sorted(input_dir.glob("*_network.parquet")):
        try:
            process_file(f, f)  # overwrite in place
        except ValueError as e:
            print(f"SKIPPED {f.name}: {e}")

def main(config):
    

    ROAD_BRIDGE_COLUMN = "bridge"
    ROAD_PAVED_COLUMN = "paved"

    RAIL_GAUGE_COLUMN = "gauge"
    RAIL_STRUCTURE_COLUMN = "structure"
    RAIL_STATUS_COLUMN = "status"
    MGR_GAUGE_THRESHOLD = 1067

    PORT_CATEGORY_COLUMN = "type"

    global PORT_CATEGORY_MAP  # so assign_maritime_port_cost_type() can see it
    PORT_CATEGORY_MAP = {
        "break": "other",
        "bulk": "dry bulk",
        "container": "storage",
        "industry": "other",
        "land": "other",
        "mixed": "general cargo",
        "oil/gas": "refinery",
        "other": "other",
        "passenger": "other",
        "raw": "other",
        "refinery": "refinery",
        "roro": "roro",
        "scrap": "other",
        "steel": "other",
        "storage": "storage",
        "vehicles": "other",
        "warehouse": "warehouse",
        "wood": "other",
    }

    
    DISPATCH[("road", "edges")] = lambda gdf: assign_road_cost_type(
        gdf, ROAD_BRIDGE_COLUMN, ROAD_PAVED_COLUMN
    )
    DISPATCH[("railway", "edges")] = lambda gdf: assign_railway_edge_cost_type(
        gdf, RAIL_GAUGE_COLUMN, RAIL_STRUCTURE_COLUMN, RAIL_STATUS_COLUMN, MGR_GAUGE_THRESHOLD
    )
    DISPATCH[("maritime", "polygons")] = lambda gdf: assign_maritime_port_cost_type(
        gdf, PORT_CATEGORY_COLUMN
    )

    incoming = config['paths']['data']
    input_dir = os.path.join(incoming, "infrastructure")

    if Path(input_dir).is_dir():
        process_directory(input_dir)
    else:
        process_file(input_dir, input_dir)

if __name__ == '__main__':
    CONFIG = load_config()
    main(CONFIG)


    

    

    
