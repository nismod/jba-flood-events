"""Expected annual values and exceedance values from an event set.

The simulated event set represents `n_years` years. With no event-to-year
mapping, each event is treated as occurring at a rate of 1 / n_years per
year, and events as independent (Poisson):

- Expected annual value (e.g. EAD) = sum of event values / n_years. This is
  exact however events are grouped into years.
- Exceedance: the k-th largest event value is exceeded or equalled k times in
  n_years, a frequency of k / n_years per year. The annual probability of at
  least one such event is 1 - exp(-k / n_years). This is an occurrence
  exceedance probability: it describes the largest single event in a year,
  not the sum of a year's events.
"""

import math

import duckdb
import pandas as pd


def exceedance_ranks(probabilities, n_years: float) -> dict[float, int]:
    """Rank k (1 = largest event) whose annual exceedance probability is each p."""
    ranks = {}
    for p in probabilities:
        if not 0 < p < 1:
            raise ValueError(f"Exceedance probability must be in (0, 1): {p}")
        frequency = -math.log(1 - p)
        ranks[p] = max(1, math.floor(frequency * n_years + 1e-9))
    return ranks


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def expected_annual(con: duckdb.DuckDBPyConnection, relation: str, keys, values, n_years: float) -> pd.DataFrame:
    """Sum of each value over events, divided by n_years, per key.

    `relation` is a table, view or subquery with one row per (key, event).
    """
    key_sql = ", ".join(_quote(k) for k in keys)
    value_sql = ", ".join(f"sum({_quote(v)}) / {float(n_years)} AS {_quote('expected_annual_' + v)}" for v in values)
    group = f"GROUP BY {key_sql} ORDER BY {key_sql}" if keys else ""
    select_keys = f"{key_sql}, " if keys else ""
    return con.sql(f"SELECT {select_keys}{value_sql} FROM {relation} {group}").df()


def exceedance(
    con: duckdb.DuckDBPyConnection, relation: str, keys, values, n_years: float, probabilities
) -> pd.DataFrame:
    """Value of each of `values` at each annual exceedance probability, per key.

    Returns one row per (key, probability) with `return_period` = 1 / p and a
    column per value. A value is 0 where fewer events than the rank have a
    positive value.
    """
    ranks = exceedance_ranks(probabilities, n_years)
    rank_table = pd.DataFrame({"probability": list(ranks), "rn": list(ranks.values())})
    rank_table["return_period"] = 1 / rank_table.probability
    con.register("_exceedance_ranks", rank_table)
    key_sql = ", ".join(_quote(k) for k in keys)
    partition = f"PARTITION BY {key_sql}" if keys else ""
    select_keys = f"{key_sql}, " if keys else ""
    rank_list = ", ".join(str(k) for k in sorted(set(ranks.values())))

    # every key that appears, crossed with every probability
    if keys:
        out = con.sql(
            f"SELECT k.*, r.probability, r.return_period, r.rn FROM "
            f"(SELECT DISTINCT {key_sql} FROM {relation}) k CROSS JOIN _exceedance_ranks r"
        ).df()
    else:
        out = rank_table.copy()

    for value in values:
        v = _quote(value)
        ranked = con.sql(
            f"SELECT {select_keys}rn, {v} FROM ("
            f"  SELECT {select_keys}{v}, row_number() OVER ({partition} ORDER BY {v} DESC) AS rn"
            f"  FROM {relation} WHERE {v} > 0"
            f") WHERE rn IN ({rank_list})"
        ).df()
        out = out.merge(ranked, on=[*keys, "rn"], how="left")
        out[value] = out[value].fillna(0.0)
    con.unregister("_exceedance_ranks")
    return out.drop(columns="rn").sort_values([*keys, "probability"], ascending=[True] * len(keys) + [False]).reset_index(drop=True)
