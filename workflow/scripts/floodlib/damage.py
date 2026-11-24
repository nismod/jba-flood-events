"""Unit costs and damage curves per asset layer."""

import os
from glob import glob

import numpy as np
import pandas as pd
import snail.damages as sndam

# cost table column holding the unit cost, by geometry kind
COST_COLUMNS = {
    "line": "mean_cost_usd_per_m",
    "polygon": "mean_cost_usd_per_sqm",
    "point": "mean_cost_usd",
}


def load_unit_costs(cost_table_path, kind: str) -> pd.Series:
    """Unit cost indexed by cost_type."""
    column = COST_COLUMNS[kind]
    cost = pd.read_csv(cost_table_path)
    missing = {"cost_type", column} - set(cost.columns)
    if missing:
        raise ValueError(f"{cost_table_path}: missing columns {sorted(missing)}")
    if cost.cost_type.duplicated().any():
        dupes = cost.cost_type[cost.cost_type.duplicated()].unique()
        raise ValueError(f"{cost_table_path}: duplicate cost_type rows {list(dupes)}")
    return cost.set_index("cost_type")[column]


def load_curves(curves_dir) -> dict:
    """Damage curve per cost_type, read from `{curves_dir}/{cost_type}.csv`."""
    curves = {}
    for path in sorted(glob(os.path.join(curves_dir, "*.csv"))):
        cost_type = os.path.splitext(os.path.basename(path))[0]
        curves[cost_type] = sndam.PiecewiseLinearDamageCurve.from_csv(
            path,
            intensity_col="flood_depth_m",
            damage_col="damage_fraction_mean",
            comment="#",
        )
    return curves


def check_costs_and_curves(cost_types, unit_costs: pd.Series, curves: dict, layer: str):
    """Raise if any costed asset has no unit cost or no damage curve."""
    needed = set(pd.Series(cost_types).dropna().unique())
    no_cost = sorted(needed - set(unit_costs.index))
    no_curve = sorted(needed - set(curves))
    if no_cost or no_curve:
        raise ValueError(
            f"{layer}: cost_types without a unit cost: {no_cost}; without a damage curve: {no_curve}"
        )


def damage_fraction(cost_type: np.ndarray, depth: np.ndarray, curves: dict) -> np.ndarray:
    """Damage fraction for each (cost_type, depth) pair; NaN where cost_type is null."""
    out = np.full(len(depth), np.nan)
    cost_type = pd.Series(cost_type)
    for ct, idx in cost_type.groupby(cost_type, dropna=True).groups.items():
        idx = np.asarray(idx)
        out[idx] = curves[ct].damage_fraction(depth[idx])
    return out
