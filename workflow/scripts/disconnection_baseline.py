"""Baseline for the population-disconnected and detour metrics (see `jba_events.disconnection`).

Builds the road network with snkit, blends population cells and urban
centres into it, and routes every node to its nearest urban centre. Writes,
to `--output_dir`:

    nodes.parquet      node, lon, lat, kind (road|population|centre), population,
                       adm1, adm0, centre_id, distance_m, parent, nearest_centre_id,
                       is_source, tin, subtree_size, status (population nodes)
    edges.parquet      u, v, asset_id (null for connectors), length_m, geometry (GeoParquet)
    sources.parquet    node, centre_id
    summary.json       population by status, and network counts

Example:

    python workflow/scripts/disconnection_baseline.py \
        --network processed_data/prepared/assets/road_edges.parquet \
        --population processed_data/population/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.tif \
        --centres processed_data/population/GHS_UCDB.gpkg --centre_id_column ID_UC_G0 \
        --admin_path processed_data/admin/adm1.gpkg --adm1_column GID_1 --adm0_column GID_0 \
        --network_crs ESRI:102023 --max_snap_m 5000 --node_tolerance_m 1 \
        --output_dir processed_data/disconnection/baseline
"""

import json
import logging
import os

import click
import geopandas as gpd
import numpy as np
from shapely.geometry import box

from jba_events.disconnection import (
    KIND_CENTRE,
    blend_network,
    build_baseline,
    centre_sources,
    read_population_cells,
)


@click.command()
@click.option("--network", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--population", "population_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--centres", "centres_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--centre_id_column", required=True)
@click.option("--admin_path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--adm1_column", default=None)
@click.option("--adm0_column", default=None)
@click.option("--network_crs", required=True, help="Metric CRS for snapping, e.g. ESRI:102023")
@click.option("--max_snap_m", required=True, type=float, help="Furthest a cell or centre may be from a road")
@click.option("--node_tolerance_m", default=1.0, show_default=True, type=float, help="Join road ends this close")
@click.option("--output_dir", required=True, type=click.Path(file_okay=False))
def main(network, population_path, centres_path, centre_id_column, admin_path, adm1_column, adm0_column,
         network_crs, max_snap_m, node_tolerance_m, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    roads = gpd.read_parquet(network, columns=["id", "geometry"])
    minx, miny, maxx, maxy = roads.to_crs("EPSG:4326").total_bounds
    # population cells up to the snap distance beyond the network are counted (as too far)
    margin = max_snap_m / 100_000
    bounds = (minx - margin, max(miny - margin, -90), maxx + margin, min(maxy + margin, 90))
    area = gpd.GeoSeries([box(*bounds)], crs="EPSG:4326")

    centres = gpd.read_file(centres_path, columns=[centre_id_column], bbox=area)
    admin = gpd.read_file(admin_path, columns=[adm1_column, adm0_column], bbox=area) if admin_path else None
    cells = read_population_cells(population_path, bounds, admin, adm1_column, adm0_column)

    nodes, edges = blend_network(roads, cells, centres, centre_id_column, network_crs, max_snap_m, node_tolerance_m)
    del roads
    sources = centre_sources(nodes, centres, centre_id_column)
    if sources.empty:
        raise ValueError("No urban centre is on or within the snap distance of the network")
    baseline = build_baseline(nodes, edges.drop(columns="geometry"), sources)

    baseline.to_parquet(os.path.join(output_dir, "nodes.parquet"), index=False)
    gpd.GeoDataFrame(edges, geometry="geometry", crs="EPSG:4326").to_parquet(
        os.path.join(output_dir, "edges.parquet"), index=False
    )
    sources.to_parquet(os.path.join(output_dir, "sources.parquet"), index=False)

    pop = baseline[baseline.status.notna()]
    summary = {status: float(v) for status, v in pop.groupby("status").population.sum().items()}
    summary["total"] = float(pop.population.sum())
    summary["network"] = {
        "nodes": len(baseline),
        "edges": len(edges),
        "connectors": int(edges.asset_id.isna().sum()),
        "sources": len(sources),
        "centres": len(centres),
        "centres_on_network": int(sources.centre_id.nunique()),
        "centres_blended": int(((baseline.kind == KIND_CENTRE) & np.isfinite(baseline.distance_m)).sum()),
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    logging.info("Baseline: %s", summary)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
