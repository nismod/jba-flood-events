
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from floodlib.events import haz_event_rp, link_haz_op
from floodlib.grid import HazardGrid
from floodlib.split import piece_sizes, split_on_window


def test_link_haz_op_within_nearest_and_unlinked():
    haz = gpd.GeoDataFrame(
        {"T500_ID": ["A", "B", "C"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(10, 0, 11, 1)],
        crs="EPSG:4326",
    )
    # op 1 in A; B has none inside but op 2 is ~55 km from its edge; C is far from everything
    ops = gpd.GeoDataFrame({"op.id": [1, 2]}, geometry=[Point(0.5, 0.5), Point(2.5, 0.5)], crs="EPSG:4326")
    links = link_haz_op(haz, ops, "T500_ID", 100_000, "ESRI:102023").set_index("haz_id")
    assert links.loc["A", "op.id"] == 1 and links.loc["A", "link"] == "within"
    assert links.loc["B", "op.id"] == 2 and links.loc["B", "link"] == "nearest"
    assert 50_000 < links.loc["B", "distance_m"] < 60_000
    assert "C" not in links.index


def test_haz_event_rp_geometric_mean():
    links = pd.DataFrame({"haz_id": ["A", "A", "B"], "op.id": [1, 2, 3]})
    events = pd.DataFrame({"event.id": [7, 7, 7], "op.id": [1, 2, 3], "rp": [10.0, 1000.0, 50.0]})
    out = haz_event_rp(events, links).set_index("haz_id")
    assert out.loc["A", "rp"] == pytest.approx(100.0)
    assert out.loc["B", "rp"] == pytest.approx(50.0)


@pytest.fixture
def grid():
    return HazardGrid(crs="EPSG:4326", width=100, height=60, transform=(0.01, 0, 10.0, 0, -0.01, 1.0))


def test_window_and_mask(grid):
    window = grid.window_for_bounds((10.103, 0.5, 10.2, 0.6))
    assert (window.col_off, window.row_off, window.width, window.height) == (10, 40, 10, 10)
    mask = window.mask([box(10.0, 0.0, 10.155, 1.0)])
    # centres at x = 10.105 .. 10.195; only those < 10.155 are inside
    assert mask[0].tolist() == [True] * 5 + [False] * 5


def test_split_lines_conserve_length_and_partition(grid):
    line = LineString([(10.005, 0.705), (10.995, 0.705)])
    gdf = gpd.GeoDataFrame({"id": ["r1"]}, geometry=[line], crs="EPSG:4326")
    window = grid.window_for_bounds((10, 0.4, 11, 1))
    pieces = split_on_window(gdf, window, "line")
    assert len(pieces) == 100
    assert sorted(pieces.i.tolist()) == list(range(100))
    assert pieces.geometry.length.sum() == pytest.approx(line.length)  # planar, in degrees
    # a window only covering part of the line keeps only that part
    half = grid.window_for_bounds((10.0, 0.4, 10.5, 1.0))
    left = split_on_window(gdf, half, "line")
    assert left.geometry.length.sum() == pytest.approx(0.495)


def test_split_polygons(grid):
    poly = box(10.48, 0.6, 10.53, 0.62)
    gdf = gpd.GeoDataFrame({"id": ["f1"]}, geometry=[poly], crs="EPSG:4326")
    window = grid.window_for_bounds((10, 0.4, 11, 1))
    pieces = split_on_window(gdf, window, "polygon")
    assert len(pieces) == 10  # 5 columns x 2 rows
    assert pieces.geometry.to_crs(3857).area.sum() == pytest.approx(gdf.to_crs(3857).area.sum())
    assert set(pieces.i) == {48, 49, 50, 51, 52}
    assert set(pieces.j) == {38, 39}
    sizes = piece_sizes(pieces, "polygon")
    # about 5.5 km x 2.2 km near the equator
    assert sizes.sum() == pytest.approx(0.05 * 0.02 * 111_320**2, rel=0.01)
