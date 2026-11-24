"""Unit costs and damage curves per asset layer."""

import os
from glob import glob

import numpy as np
import pandas as pd
import snail.damages as sndam

from jba_events.depth import event_depths, zero_depth_rp

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


# Cap the size of the (pieces x events) arrays computed at once
MAX_CELLS_PER_CHUNK = 5_000_000


class PieceCosting:
    """Per-piece inputs to the damage calculation, for one layer in one HAZ.

    `splits` is a piece table from `jba_events.exposure.split_layer`. Only
    pieces that are wet at some mapped RP are kept.
    """

    def __init__(self, splits: pd.DataFrame, rps, horp: float, kind: str, layer: str,
                 cost_table=None, curves_dir=None):
        self.rps = np.array(sorted(rps), dtype=float)
        depth_cols = [f"depth_rp{rp:g}" for rp in self.rps]
        depths = splits[depth_cols].to_numpy(dtype=np.float64)
        wet = (depths > 0).any(axis=1)
        self.splits = splits[wet].reset_index(drop=True)
        self.depths = depths[wet]
        self.first_wet, self.z = zero_depth_rp(self.depths, self.rps, horp)
        self.size = self.splits["size"].to_numpy(dtype=float)
        self.sop = np.nan_to_num(self.splits["sop"].to_numpy(dtype=float), nan=-np.inf)
        self.cost_type = self.splits["cost_type"].to_numpy(dtype=object)
        self.costed = pd.notna(self.cost_type)
        self.unit_cost = np.full(len(self.splits), np.nan)
        self.curves = {}
        if self.costed.any():
            if cost_table is None or not os.path.exists(cost_table):
                raise ValueError(f"{layer}: pieces have a cost_type but there is no cost table ({cost_table})")
            unit_costs = load_unit_costs(cost_table, kind)
            self.curves = load_curves(curves_dir) if curves_dir and os.path.isdir(curves_dir) else {}
            check_costs_and_curves(self.cost_type, unit_costs, self.curves, layer)
            self.unit_cost = self.splits["cost_type"].map(unit_costs).to_numpy(dtype=float)

    def __len__(self):
        return len(self.splits)

    def event_chunks(self, events: pd.DataFrame):
        """Yield slices of `events` sized to bound memory use."""
        step = max(1, MAX_CELLS_PER_CHUNK // max(1, len(self)))
        for start in range(0, len(events), step):
            yield events.iloc[start : start + step]

    def matrices(self, event_rp: np.ndarray):
        """Depth, damage fraction, damage and protection, each (pieces, events).

        Damage is NaN for exposure-only pieces (no cost_type).
        """
        d = event_depths(self.depths, self.rps, self.first_wet, self.z, event_rp)
        frac = np.full(d.shape, np.nan)
        for ct in pd.unique(self.cost_type[self.costed]):
            rows = np.flatnonzero(self.cost_type == ct)
            frac[rows] = np.asarray(self.curves[ct].damage_fraction(d[rows].ravel())).reshape(len(rows), -1)
        damage = (self.size * self.unit_cost)[:, None] * frac
        protected = event_rp[None, :] <= self.sop[:, None]
        return d, frac, damage, protected


def piece_event_table(costing: PieceCosting, events: pd.DataFrame, event_id_col: str) -> pd.DataFrame:
    """One row per (piece, event) with depth > 0.

    This is the full-detail table. The workflow doesn't store it, because it
    is a function of the piece table and the event RPs; call this to rebuild
    it for a subset of events when piece-level detail is needed.
    """
    frames = []
    for ev in costing.event_chunks(events):
        event_rp = ev["rp"].to_numpy(dtype=float)
        d, frac, damage, protected = costing.matrices(event_rp)
        p, e = np.nonzero(d > 0)
        frames.append(
            pd.DataFrame(
                {
                    "split_id": costing.splits["split_id"].to_numpy()[p],
                    "id": costing.splits["id"].to_numpy()[p],
                    "event_id": ev[event_id_col].to_numpy()[e],
                    "rp": event_rp[e].astype(np.float32),
                    "depth_m": d[p, e].astype(np.float32),
                    "protected": protected[p, e],
                    "damage_fraction": frac[p, e].astype(np.float32),
                    "damage_undefended_usd": damage[p, e],
                    "damage_defended_usd": np.where(protected[p, e] & ~np.isnan(damage[p, e]), 0.0, damage[p, e]),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


ASSET_EVENT_COLUMNS = [
    "id",
    "adm1",
    "event_id",
    "rp",
    "max_depth_m",
    "max_depth_defended_m",
    "exposed_size",
    "exposed_size_defended",
    "damage_undefended_usd",
    "damage_defended_usd",
]


def asset_event_chunks(costing: PieceCosting, events: pd.DataFrame, event_id_col: str):
    """Yield tables with one row per (asset, adm1, event) with depth > 0.

    Pieces are summed to the asset within each admin-1 region, so an asset
    that crosses a boundary has one row per region. "Defended" values treat
    a piece as dry when the event RP is at or below its standard of protection.
    Damage is null for assets with no costed piece.
    """
    if len(costing) == 0 or len(events) == 0:
        return
    keys = costing.splits[["id", "adm1"]]
    group = keys.groupby(["id", "adm1"], dropna=False, sort=False).ngroup().to_numpy()
    order = np.argsort(group, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(group[order]) != 0])
    group_keys = keys.iloc[order[starts]].reset_index(drop=True)
    group_costed = np.add.reduceat(costing.costed[order].astype(int), starts) > 0
    size = costing.size[order][:, None]

    for ev in costing.event_chunks(events):
        event_rp = ev["rp"].to_numpy(dtype=float)
        d, _, damage, protected = costing.matrices(event_rp)
        d, protected = d[order], protected[order]
        damage = np.nan_to_num(damage[order], nan=0.0)
        d_def = np.where(protected, 0.0, d)

        max_d = np.maximum.reduceat(d, starts, axis=0)
        g, e = np.nonzero(max_d > 0)
        costed = group_costed[g]

        def pick(matrix, how):
            return how.reduceat(matrix, starts, axis=0)[g, e]

        dmg = np.where(costed, pick(damage, np.add), np.nan)
        dmg_def = np.where(costed, pick(np.where(protected, 0.0, damage), np.add), np.nan)
        yield pd.DataFrame(
            {
                "id": group_keys["id"].to_numpy()[g],
                "adm1": group_keys["adm1"].to_numpy()[g],
                "event_id": ev[event_id_col].to_numpy()[e],
                "rp": event_rp[e].astype(np.float32),
                "max_depth_m": max_d[g, e].astype(np.float32),
                "max_depth_defended_m": pick(d_def, np.maximum).astype(np.float32),
                "exposed_size": pick(size * (d > 0), np.add),
                "exposed_size_defended": pick(size * (d_def > 0), np.add),
                "damage_undefended_usd": dmg,
                "damage_defended_usd": dmg_def,
            }
        )
