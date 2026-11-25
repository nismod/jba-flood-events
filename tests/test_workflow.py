"""Run the whole Snakefile on the synthetic dataset and check the outputs."""

import json
import math
import os
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pyproj import Geod

from scipy.sparse.csgraph import dijkstra

from jba_events.damage import PieceCosting, asset_event_chunks, piece_event_table
from jba_events.disconnection import min_csr

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
YEARS = 4  # risk.years in the test config


@pytest.fixture(scope="module")
def run(dataset):
    root, data = dataset
    env = dict(os.environ, PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])
    cmd = [
        sys.executable, "-m", "snakemake", "--cores", "2", "all", "--quiet",
        "--configfile", "workflow/config.yaml", os.path.join(root, "test_config.yaml"),
        "--config", f"paths_file={os.path.join(root, 'config.json')}",
    ]
    result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-5000:]
    return data


def read(data, *parts):
    return pd.read_parquet(os.path.join(data, *parts))


def test_haz_links_and_event_rps(run):
    links = read(run, "prepared", "events", "haz_op_links.parquet")
    assert sorted(links.haz_id.unique()) == ["A", "B"]  # C is too far from any OP
    rp = read(run, "prepared", "events", "Obs_haz_event_rp.parquet").set_index(["haz_id", "event.id"])
    assert rp.loc[("A", 1), "rp"] == pytest.approx(30)
    assert rp.loc[("B", 1), "rp"] == pytest.approx(200)  # gmean(100, 400)


def test_splits_partition_assets_between_hazs(run):
    splits = gpd.read_parquet(os.path.join(run, "exposure", "road_edges"))
    r1 = splits[splits.id == "r1"]
    # every cell of the road appears exactly once across the two HAZs
    assert len(r1) == 100
    assert not r1.duplicated(["raster_i", "raster_j"]).any()
    assert set(r1[r1.haz == "A"].raster_i) == set(range(0, 50))
    assert set(r1[r1.haz == "B"].raster_i) == set(range(50, 100))
    assert r1["size"].sum() == pytest.approx(0.99 * 111_320 * np.cos(np.radians(0.705)), rel=0.005)
    # defended pieces take the highest overlapping SoP
    defended = r1[r1.sop.notna()]
    assert set(defended.sop) == {100.0}
    assert set(defended.raster_i) == set(range(60, 70))
    # admin-1 boundary at x = 10.3 is column 30
    assert set(r1[r1.adm1 == "X1"].raster_i) == set(range(0, 30))
    assert set(r1[r1.adm1 == "Y1"].raster_i) == set(range(30, 100))


def test_asset_event_damages(run):
    dmg = read(run, "asset_event_damages", "Obs", "road_edges")
    assert not dmg.duplicated(["haz", "id", "adm1", "event_id"]).any()

    # event 1 in HAZ A has RP 30: only cells wet from RP20 (columns 40-49) flood, all in Y1
    a = dmg[(dmg.haz == "A") & (dmg.event_id == 1) & (dmg.id == "r1")]
    assert a.adm1.tolist() == ["Y1"]
    depth = 0.1 + 0.1 * (np.log(30) - np.log(20)) / (np.log(50) - np.log(20))
    assert a.max_depth_m.iloc[0] == pytest.approx(depth, abs=1e-6)
    splits = read(run, "exposure", "road_edges")
    wet = splits[(splits.haz == "A") & (splits.id == "r1") & (splits.raster_i >= 40)]
    assert a.exposed_size.iloc[0] == pytest.approx(wet["size"].sum())
    # curve is 0.5 damage per metre of depth up to 1 m; cost 100 USD/m
    assert a.damage_undefended_usd.iloc[0] == pytest.approx(wet["size"].sum() * 100 * 0.5 * depth, rel=1e-5)

    # HAZ B event 1 has RP 200 > SoP 100: nothing is protected
    b = dmg[(dmg.haz == "B") & (dmg.event_id == 1) & (dmg.id == "r1")]
    assert (b.damage_defended_usd == b.damage_undefended_usd).all()


def test_defended_protection_below_sop(run):
    dmg = read(run, "asset_event_damages", "Sim", "road_edges")
    b = dmg[(dmg.haz == "B") & (dmg.event_id == 11) & (dmg.id == "r1")].iloc[0]  # RP 50 <= SoP 100
    splits = read(run, "exposure", "road_edges")
    protected = splits[(splits.haz == "B") & splits.sop.notna()]
    assert b.exposed_size - b.exposed_size_defended == pytest.approx(protected["size"].sum())
    assert b.damage_defended_usd == pytest.approx(b.damage_undefended_usd * b.exposed_size_defended / b.exposed_size)
    # Sim event 10 in A is capped at the RP1500 depth
    a = dmg[(dmg.haz == "A") & (dmg.event_id == 10)]
    assert np.allclose(a.max_depth_m, 0.6)


def test_asset_table_matches_piece_table(run):
    """The stored asset table is the piece-by-event table summed to assets."""
    splits = read(run, "exposure", "road_edges", "haz=B", "splits.parquet")
    events = read(run, "prepared", "events", "Sim_haz_event_rp.parquet")
    events = events[events.haz_id == "B"]
    costing = PieceCosting(
        splits, [20, 50, 100, 200, 500, 1500], 11, "line", "road_edges",
        cost_table=os.path.join(run, "costs", "road_edges.csv"), curves_dir=os.path.join(run, "curves", "road"),
    )
    pieces = piece_event_table(costing, events, "event.id")
    assert not pieces.duplicated(["split_id", "event_id"]).any()
    assets = pd.concat(asset_event_chunks(costing, events, "event.id"))
    for column in ["damage_undefended_usd", "damage_defended_usd"]:
        assert assets[column].sum() == pytest.approx(pieces[column].sum())
    assert assets.max_depth_m.max() == pytest.approx(pieces.depth_m.max())


def test_exposure_only_layers(run):
    nodes = read(run, "asset_event_damages", "Obs", "road_nodes")
    # n2 lies in an always-dry column
    assert nodes.id.tolist() == ["n1"] and nodes.damage_undefended_usd.isna().all()
    assert nodes.exposed_size.tolist() == [1.0]
    # r2 has no cost_type: exposure-only rows in the costed layer
    edges = read(run, "asset_event_damages", "Obs", "road_edges")
    assert edges[edges.id == "r2"].damage_undefended_usd.isna().all()


def test_empty_haz_outputs(run):
    # HAZ C is off the grid and has no OP: outputs exist and are empty
    assert len(gpd.read_parquet(os.path.join(run, "exposure", "road_edges", "haz=C", "splits.parquet"))) == 0
    assert len(read(run, "asset_event_damages", "Obs", "road_edges", "haz=C", "part.parquet")) == 0


def test_event_losses_by_region(run):
    assets = read(run, "asset_event_damages", "Sim", "road_edges")
    total = assets.damage_undefended_usd.sum()
    for level in ["haz", "adm1", "adm0"]:
        losses = read(run, "event_losses", "Sim", "road_edges", f"{level}.parquet")
        assert losses.damage_undefended_usd.sum() == pytest.approx(total)
    adm0 = read(run, "event_losses", "Sim", "road_edges", "adm0.parquet")
    # event 11 floods country Y in both HAZ A and HAZ B: one summed row
    assert len(adm0[(adm0.adm0 == "Y") & (adm0.event_id == 11)]) == 1


def test_expected_annual_and_exceedance(run):
    assets = read(run, "asset_event_damages", "Sim", "road_edges")
    r1 = assets[assets.id == "r1"]
    ead = read(run, "risk", "road_edges", "asset_expected_annual.parquet").set_index("id")
    assert ead.loc["r1", "expected_annual_damage_undefended_usd"] == pytest.approx(r1.damage_undefended_usd.sum() / YEARS)
    assert ead.loc["r1", "expected_annual_damage_defended_usd"] < ead.loc["r1", "expected_annual_damage_undefended_usd"]
    assert np.isnan(ead.loc["r2", "expected_annual_damage_undefended_usd"])

    # with 4 years: p = 0.2 is the largest event loss, p = 0.5 the second largest
    per_event = r1.groupby("event_id").damage_undefended_usd.sum().sort_values(ascending=False)
    exc = read(run, "risk", "road_edges", "asset_exceedance.parquet")
    exc = exc[exc.id == "r1"].set_index("probability")
    assert exc.loc[0.2, "damage_undefended_usd"] == pytest.approx(per_event.iloc[0])
    assert exc.loc[0.5, "damage_undefended_usd"] == pytest.approx(per_event.iloc[1])
    assert exc.loc[0.2, "return_period"] == pytest.approx(5)

    # all-layer totals are the sum of the layers
    layers = ["road_edges", "road_nodes", "airport_field_polygons"]
    by_layer = sum(
        read(run, "risk", layer, "adm0_expected_annual.parquet").expected_annual_damage_undefended_usd.sum()
        for layer in layers
    )
    total = read(run, "risk", "all", "adm0_expected_annual.parquet").expected_annual_damage_undefended_usd.sum()
    assert total == pytest.approx(by_layer)


def test_qa_summary(run):
    qa = pd.read_csv(os.path.join(run, "qa", "exposure_summary.csv")).set_index(["layer", "haz"])
    splits = read(run, "exposure", "road_edges")
    assert qa.loc[("road_edges", "A"), "pieces"] == (splits.haz == "A").sum()
    assert qa.loc[("road_edges", "A"), "assets"] == splits[splits.haz == "A"].id.nunique()
    assert qa.loc[("road_edges", "B"), "size_defended"] > 0
    assert qa.loc[("road_edges", "A"), "size_wet_rp20"] < qa.loc[("road_edges", "A"), "size_wet_rp50"]


def test_population_baseline(run):
    with open(os.path.join(run, "disconnection", "baseline", "summary.json")) as fh:
        summary = json.load(fh)
    assert {k: summary[k] for k in ["served", "no_route", "too_far", "total"]} == {
        "served": 190.0, "no_route": 9.0, "too_far": 7.0, "total": 206.0,
    }
    nodes = read(run, "disconnection", "baseline", "nodes.parquet")
    edges = read(run, "disconnection", "baseline", "edges.parquet")
    # every blended point has exactly one connector, and split pieces keep their road's asset id
    blended = nodes[nodes.kind.isin(["population", "centre"]) & (nodes.status != "too_far")]
    connectors = edges[edges.asset_id.isna()]
    assert len(connectors) == len(blended)
    assert set(edges.asset_id.dropna()) == {"r1", "r2", "r3"}
    roads = gpd.read_parquet(os.path.join(run, "infrastructure", "africa_road_edges_network.parquet"))
    geod = Geod(ellps="WGS84")
    for asset in ["r1", "r3"]:
        assert edges[edges.asset_id == asset].length_m.sum() == pytest.approx(
            geod.geometry_length(roads[roads.id == asset].geometry.iloc[0]), rel=1e-6
        )
    # people inside the urban centre are at a source
    inside = nodes[(nodes.kind == "population") & (nodes.population == 50)]
    assert inside.is_source.all() and (inside.distance_m == 0).all()


def test_population_disconnected_and_detoured(run):
    events = pd.concat(read(run, "disconnection", "Sim", "events", f"part-{i}.parquet") for i in range(2))
    # only Sim event 10 floods r1 to 0.5 m or more
    assert set(events.event_id) == {10}
    undefended = events[events.protection == "undefended"].set_index("adm1")
    assert undefended.index.tolist() == ["X1"]
    # the 100 on r1 are cut off; the 40 on the bypass go the long way round
    assert undefended.loc["X1", "population_disconnected"] == 100
    assert undefended.loc["X1", "population_detoured"] == 40

    # detour: checked against a full search of the network without r1
    nodes = read(run, "disconnection", "baseline", "nodes.parquet")
    edges = read(run, "disconnection", "baseline", "edges.parquet")
    sources = read(run, "disconnection", "baseline", "sources.parquet").node.to_numpy()
    intact = edges[edges.asset_id != "r1"]
    u, v, w = intact.u.to_numpy(), intact.v.to_numpy(), intact.length_m.to_numpy()
    after = dijkstra(min_csr(np.r_[u, v], np.r_[v, u], np.r_[w, w], len(nodes)), indices=sources, min_only=True)
    person = nodes[(nodes.kind == "population") & (nodes.population == 40)].iloc[0]
    detour_km = (after[person.node] - person.distance_m) / 1000
    assert undefended.loc["X1", "detour_person_km"] == pytest.approx(40 * detour_km, rel=1e-6)
    # the bypass via y = 1.05 is about 0.35 degrees of extra latitude, each way
    assert detour_km == pytest.approx(0.345 * 110.6, rel=0.05)

    total = read(run, "disconnection", "Sim", "risk", "total_expected_annual.parquet").set_index("protection")
    assert total.loc["undefended", "expected_annual_population_disconnected"] == pytest.approx(100 / YEARS)
    assert total.loc["undefended", "expected_annual_population_detoured"] == pytest.approx(40 / YEARS)
    assert total.loc["undefended", "annual_probability_any"] == pytest.approx(1 - math.exp(-1 / YEARS))
    exc = read(run, "disconnection", "Sim", "risk", "adm0_exceedance.parquet")
    exc = exc[exc.protection == "undefended"].set_index("probability")
    assert exc.loc[0.2, "population_disconnected"] == 100 and exc.loc[0.5, "population_disconnected"] == 0


def test_network_criticality(run):
    crit = read(run, "disconnection", "criticality.parquet").set_index("asset_id")
    assert crit.index.tolist() == ["r1", "r3"]  # ranked; r2 carries no served population
    # everyone served outside the centre routes along r1; r1 alone failing is event 10's impact
    assert crit.loc["r1", "population_routed"] == 140
    assert crit.loc["r1", "population_disconnected"] == 100
    assert crit.loc["r1", "population_detoured"] == 40
    events = pd.concat(read(run, "disconnection", "Sim", "events", f"part-{i}.parquet") for i in range(2))
    assert crit.loc["r1", "detour_person_km"] == pytest.approx(
        events[events.protection == "undefended"].detour_person_km.sum()
    )
    # the 40 on the bypass have no other way out if r3 fails
    assert crit.loc["r3", "population_routed"] == 40 and crit.loc["r3", "population_disconnected"] == 40
    # r1 fails in one Sim event in 4 years; the bypass never floods
    assert crit.loc["r1", "annual_failure_frequency_undefended"] == pytest.approx(1 / YEARS)
    assert crit.loc["r3", "annual_failure_frequency_undefended"] == 0
    assert crit.loc["r1", "expected_annual_population_disconnected_if_alone"] == pytest.approx(100 / YEARS)
