"""Interpolate flood depth for an event return period from mapped RP depths.

For mapped return periods r_1 < ... < r_n with depths d_1..d_n at a cell:

- Let r_f be the first mapped RP with depth > 0. The zero-depth RP z is the
  hazard onset return period (HORP, default 11) if r_f is the lowest mapped
  RP. Otherwise z is the mean of r_f and the mapped RP below it, e.g. 35 for a
  cell first wet at RP50.
- Depth is 0 for rp <= z.
- Between z and r_f, depth rises from 0 to d_f, linearly in log(rp).
- Between mapped RPs above r_f, depth is interpolated linearly in log(rp).
- For rp >= r_n, depth is d_n.
"""

import numpy as np


def zero_depth_rp(depths: np.ndarray, rps: np.ndarray, horp: float) -> tuple[np.ndarray, np.ndarray]:
    """Index of the first wet RP, and the zero-depth RP, for each row of `depths`.

    depths: (n_cells, n_rps), with columns in increasing RP order.
    Returns (first_wet, z). first_wet is n_rps for cells that are never wet,
    and z is inf for those cells.
    """
    rps = np.asarray(rps, dtype=float)
    wet = depths > 0
    any_wet = wet.any(axis=1)
    first_wet = np.where(any_wet, wet.argmax(axis=1), len(rps))
    below = np.concatenate([[np.nan], rps[:-1]])
    z = np.full(len(depths), np.inf)
    lowest = any_wet & (first_wet == 0)
    later = any_wet & (first_wet > 0)
    z[lowest] = horp
    z[later] = (rps[first_wet[later]] + below[first_wet[later]]) / 2
    if not (horp < rps[0]):
        raise ValueError(f"HORP ({horp}) must be below the lowest mapped RP ({rps[0]})")
    return first_wet, z


def event_depths(
    depths: np.ndarray,
    rps: np.ndarray,
    first_wet: np.ndarray,
    z: np.ndarray,
    event_rp: np.ndarray,
) -> np.ndarray:
    """Depth for each cell (rows) and event (columns).

    depths: (n_cells, n_rps) mapped depths; first_wet, z from `zero_depth_rp`.
    event_rp: (n_events,) event return periods.
    """
    rps = np.asarray(rps, dtype=float)
    event_rp = np.minimum(np.asarray(event_rp, dtype=float), rps[-1])
    # u: index of the mapped RP at or above each event RP
    u = np.searchsorted(rps, event_rp, side="left")  # (n_events,)
    u = np.minimum(u, len(rps) - 1)

    n_cells = depths.shape[0]
    d_u = depths[:, u]  # (n_cells, n_events)
    r_u = rps[u][None, :]

    # lower knot: the zero-depth point in the first wet bin, else the mapped RP below
    in_first_bin = u[None, :] == first_wet[:, None]
    r_below = np.concatenate([[np.nan], rps[:-1]])[u]
    d_below = np.where(u > 0, depths[:, np.maximum(u - 1, 0)], 0.0)
    r_l = np.where(in_first_bin, z[:, None], r_below[None, :])
    d_l = np.where(in_first_bin, 0.0, d_below)

    log_rp = np.log(event_rp)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = (log_rp - np.log(r_l)) / (np.log(r_u) - np.log(r_l))
    factor = np.clip(np.nan_to_num(factor, nan=1.0), 0.0, 1.0)
    out = d_l + (d_u - d_l) * factor

    # dry below the zero-depth RP, and in bins below the first wet RP
    dry = (event_rp[None, :] <= z[:, None]) | (u[None, :] < first_wet[:, None])
    out[dry] = 0.0
    out = np.maximum(out, 0.0)
    assert out.shape == (n_cells, len(event_rp))
    return out
