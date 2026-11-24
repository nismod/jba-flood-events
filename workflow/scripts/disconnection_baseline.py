"""Baseline for the population-disconnected metric (see `jba_events.disconnection`).

Routes every populated cell over the network to its nearest urban centre and
writes, to `--output_dir`:

    edges.parquet        u, v, id, length_m
    nodes.parquet        node, lon, lat, distance_m, centre_node, centre_id
    sources.parquet      node, centre_id (the nodes standing for each urban centre)
    cells.parquet        lon, lat, population, adm1, adm0, node, snap_distance_m, status
    population.parquet   node, adm1, adm0, population (served population only)
    summary.json         population totals by status

Example:

    python workflow/scripts/disconnection_baseline.py \
        --network processed_data/prepared/assets/road_edges.parquet \
        --population processed_data/population/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.tif \
        --centres processed_data/population/GHS_UCDB.gpkg --centre_id_column ID_UC_G0 \
        --admin_path processed_data/admin/adm1.gpkg --adm1_column GID_1 --adm0_column GID_0 \
        --max_snap_m 5000 --output_dir processed_data/disconnection/baseline
"""

import json
import logging
import os

import click
import geopandas as gpd

from jba_events.disconnection import (
    assign_admin,
    build_graph,
    centre_source_nodes,
    read_population_cells,
    route_to_nearest_centre,
    served_population,
)


@click.command()
@click.option("--network", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--population", "population_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--centres", "centres_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--centre_id_column", required=True)
@click.option("--admin_path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--adm1_column", default=None)
@click.option("--adm0_column", default=None)
@click.option("--max_snap_m", required=True, type=float, help="Furthest a cell or centre may be from the network")
@click.option("--output_dir", required=True, type=click.Path(file_okay=False))
def main(network, population_path, centres_path, centre_id_column, admin_path, adm1_column, adm0_column,
         max_snap_m, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    edges = gpd.read_parquet(network, columns=["id", "geometry"])
    edge_table, nodes = build_graph(edges)
    del edges
    bounds = (nodes.lon.min(), nodes.lat.min(), nodes.lon.max(), nodes.lat.max())

    centres = gpd.read_file(centres_path, columns=[centre_id_column], bbox=bounds)
    sources = centre_source_nodes(nodes, centres, centre_id_column, max_snap_m)
    if sources.empty:
        raise ValueError("No urban centre is within the snap distance of the network")
    routes = route_to_nearest_centre(edge_table, len(nodes), sources)

    # include cells just outside the network extent; cells further out are not counted at all
    margin = max_snap_m / 100_000
    cells = read_population_cells(
        population_path,
        (bounds[0] - margin, max(bounds[1] - margin, -90), bounds[2] + margin, min(bounds[3] + margin, 90)),
    )
    admin = gpd.read_file(admin_path, columns=[adm1_column, adm0_column]) if admin_path else None
    cells[["adm1", "adm0"]] = assign_admin(cells.lon.values, cells.lat.values, admin, adm1_column, adm0_column)
    cells, population = served_population(cells, nodes, routes, max_snap_m)

    edge_table.to_parquet(os.path.join(output_dir, "edges.parquet"), index=False)
    nodes.merge(routes, on="node").to_parquet(os.path.join(output_dir, "nodes.parquet"), index=False)
    sources.to_parquet(os.path.join(output_dir, "sources.parquet"), index=False)
    cells.to_parquet(os.path.join(output_dir, "cells.parquet"), index=False)
    population.to_parquet(os.path.join(output_dir, "population.parquet"), index=False)
    summary = cells.groupby("status").population.sum().to_dict()
    summary["total"] = float(cells.population.sum())
    with open(os.path.join(output_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    logging.info("Baseline population by status: %s", summary)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
