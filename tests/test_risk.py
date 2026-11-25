import duckdb
import numpy as np
import pandas as pd
import pytest

from jba_events.risk import exceedance, exceedance_ranks, expected_annual


def test_exceedance_ranks():
    ranks = exceedance_ranks([0.01, 0.001], 10_000)
    assert ranks == {0.01: 100, 0.001: 10}
    assert exceedance_ranks([0.0001], 100) == {0.0001: 1}  # rarer than the set resolves: largest event
    with pytest.raises(ValueError):
        exceedance_ranks([1.0], 100)


def test_expected_annual_and_exceedance_by_key():
    con = duckdb.connect()
    events = pd.DataFrame(
        {"region": ["a"] * 5 + ["b"], "event_id": range(6), "loss": [5.0, 1.0, 4.0, 2.0, 3.0, 10.0]}
    )
    con.register("events", events)
    ead = expected_annual(con, "events", ["region"], ["loss"], 10).set_index("region")
    assert ead.loc["a", "expected_annual_loss"] == pytest.approx(1.5)
    # 10 years: p = 1 - exp(-0.3) is the third largest; region b has only one event
    p = 1 - np.exp(-0.3)
    exc = exceedance(con, "events", ["region"], ["loss"], 10, [p]).set_index("region")
    assert exc.loc["a", "loss"] == 3.0
    assert exc.loc["b", "loss"] == 0.0
