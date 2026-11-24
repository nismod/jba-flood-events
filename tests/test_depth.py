import numpy as np
import pytest

from floodlib.depth import event_depths, zero_depth_rp

RPS = np.array([20, 50, 100, 200, 500, 1500], dtype=float)


def depths_for(row, event_rps, horp=11):
    depths = np.array([row], dtype=float)
    first_wet, z = zero_depth_rp(depths, RPS, horp)
    return event_depths(depths, RPS, first_wet, z, np.array(event_rps, dtype=float))[0]


def test_zero_depth_rp_follows_readme():
    depths = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # wet at lowest RP -> HORP
            [0, 0.2, 0.3, 0.4, 0.5, 0.6],  # first wet at 50 -> mean(20, 50)
            [0, 0, 0, 0, 0, 0.6],  # first wet at 1500 -> mean(500, 1500)
            [0, 0, 0, 0, 0, 0],  # never wet
        ]
    )
    first_wet, z = zero_depth_rp(depths, RPS, horp=11)
    assert first_wet.tolist() == [0, 1, 5, 6]
    assert z[:3].tolist() == [11, 35, 1000]
    assert np.isinf(z[3])


def test_wet_from_lowest_rp():
    row = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    d = depths_for(row, [5, 11, 20, 50, 1500, 5000])
    assert d.tolist() == pytest.approx([0, 0, 0.1, 0.2, 0.6, 0.6])
    # log-linear between HORP and RP20
    mid = np.sqrt(11 * 20)
    assert depths_for(row, [mid])[0] == pytest.approx(0.05)
    # log-linear between mapped RPs
    assert depths_for(row, [np.sqrt(20 * 50)])[0] == pytest.approx(0.15)


def test_first_wet_at_rp50():
    row = [0, 0.2, 0.3, 0.4, 0.5, 0.6]
    d = depths_for(row, [20, 30, 35, 50, 100])
    assert d.tolist() == pytest.approx([0, 0, 0, 0.2, 0.3])
    assert depths_for(row, [np.sqrt(35 * 50)])[0] == pytest.approx(0.1)


def test_never_wet():
    assert depths_for([0] * 6, [10, 100, 10000]).tolist() == [0, 0, 0]


def test_horp_must_be_below_lowest_rp():
    with pytest.raises(ValueError):
        zero_depth_rp(np.ones((1, 6)), RPS, horp=20)
