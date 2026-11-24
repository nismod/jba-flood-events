"""Run the whole Snakefile on the synthetic dataset and check the outputs."""

import os
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def run(dataset):
    root, data = dataset
    env = dict(os.environ, PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])
    cmd = [
        sys.executable, "-m", "snakemake", "--cores", "2", "all",
        "--config", f"paths_file={os.path.join(root, 'config.json')}",
    ]
    result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-5000:]
    return data


def read_damages(data, event_type, layer):
    return pd.read_parquet(os.path.join(data, "event_damages", event_type, layer))


def test_haz_links_and_event_rps(run):
    links = pd.read_parquet(os.path.join(run, "prepared", "events", "haz_op_links.parquet"))
    assert sorted(links.haz_id.unique()) == ["A", "B"]  # C is too far from any OP
    rp = pd.read_parquet(os.path.join(run, "prepared", "events", "Obs_haz_event_rp.parquet")).set_index(["haz_id", "event.id"])
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
    # 0.99 degrees of longitude near 0.7N
    assert r1["size"].sum() == pytest.approx(0.99 * 111_320 * np.cos(np.radians(0.705)), rel=0.005)
    # defended pieces take the highest overlapping SoP
    defended = r1[r1.sop.notna()]
    assert set(defended.sop) == {100.0}
    assert set(defended.raster_i) == set(range(60, 70))


def test_event_damages(run):
    dmg = read_damages(run, "Obs", "road_edges")
    splits = pd.read_parquet(os.path.join(run, "exposure", "road_edges"), columns=["haz", "split_id", "raster_i", "size", "sop"])
    df = dmg.merge(splits, on=["haz", "split_id"])
    # one row per piece and event, no fan-out
    assert not df.duplicated(["haz", "split_id", "event_id"]).any()

    a = df[(df.haz == "A") & (df.event_id == 1)]
    # event 1 in HAZ A has RP 30: wet-from-RP20 cells interpolate between RP20 (0.1) and RP50 (0.2)
    expected = 0.1 + 0.1 * (np.log(30) - np.log(20)) / (np.log(50) - np.log(20))
    early = a[a.raster_i >= 40]
    assert np.allclose(early.depth_m, expected, atol=1e-6)
    # cells first wet at RP50 have zero depth until RP35, so are dry at RP30
    assert a.raster_i.min() == 40
    r1 = early[early.id == "r1"]
    assert np.allclose(r1.damage_undefended_usd, r1["size"] * 100 * 0.5 * r1.depth_m, rtol=1e-5)

    # HAZ B event 1 has RP 200 > SoP 100, so defended pieces are not protected
    b = df[(df.haz == "B") & (df.event_id == 1)]
    assert not b.protected.any()
    assert (b.damage_defended_usd == b.damage_undefended_usd).all()


def test_defended_protection_below_sop(run):
    dmg = read_damages(run, "Sim", "road_edges")
    splits = pd.read_parquet(os.path.join(run, "exposure", "road_edges"), columns=["haz", "split_id", "sop"])
    df = dmg.merge(splits, on=["haz", "split_id"])
    b = df[(df.haz == "B") & (df.event_id == 11)]  # RP 50 <= SoP 100
    protected = b[b.sop.notna()]
    assert len(protected) and protected.protected.all()
    assert (protected.damage_defended_usd == 0).all()
    assert (protected.damage_undefended_usd > 0).all()
    # Sim event 10 in A is capped at the RP1500 depth
    a = df[(df.haz == "A") & (df.event_id == 10)]
    assert np.allclose(a.depth_m, 0.6)


def test_exposure_only_layers(run):
    nodes = read_damages(run, "Obs", "road_nodes")
    # n2 lies in an always-dry column
    assert nodes.id.tolist() == ["n1"] and nodes.damage_undefended_usd.isna().all()
    # r2 has no cost_type: exposure-only rows in the costed layer
    edges = read_damages(run, "Obs", "road_edges")
    assert edges[edges.id == "r2"].damage_undefended_usd.isna().all()


def test_polygon_damages(run):
    dmg = read_damages(run, "Obs", "airport_field_polygons")
    splits = pd.read_parquet(os.path.join(run, "exposure", "airport_field_polygons"))
    assert splits.groupby("haz").size().to_dict() == {"A": 4, "B": 6}
    assert dmg.damage_undefended_usd.gt(0).all()


def test_empty_haz_outputs(run):
    # HAZ C is off the grid and has no OP: outputs exist and are empty
    splits = gpd.read_parquet(os.path.join(run, "exposure", "road_edges", "haz=C", "splits.parquet"))
    assert len(splits) == 0
    dmg = pd.read_parquet(os.path.join(run, "event_damages", "Obs", "road_edges", "haz=C", "part.parquet"))
    assert len(dmg) == 0
