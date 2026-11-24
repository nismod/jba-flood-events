"""Link observation points (OPs) to HAZs, and derive a return period per HAZ and event.

Rules from the JBA technical notes (see README):

- An OP inside a HAZ is linked to that HAZ.
- A HAZ with no OP inside is linked to the OP nearest its boundary, if that
  OP is within `max_distance_m`. Otherwise the HAZ is not at risk.
- If a HAZ has several OPs, the event RP for the HAZ is the geometric mean of
  the RPs of its OPs in that event.
"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

HAZ_ID = "haz_id"
OP_ID = "op.id"
EVENT_ID = "event.id"


def link_haz_op(
    haz: gpd.GeoDataFrame,
    ops: gpd.GeoDataFrame,
    haz_id_col: str,
    max_distance_m: float,
    distance_crs: str,
) -> pd.DataFrame:
    """Table of (haz_id, op.id, link, distance_m)."""
    haz = haz[[haz_id_col, "geometry"]].rename(columns={haz_id_col: HAZ_ID})
    ops = ops[[OP_ID, "geometry"]].to_crs(haz.crs)

    within = gpd.sjoin(ops, haz, predicate="within", how="inner")[[HAZ_ID, OP_ID]]
    within["link"] = "within"
    within["distance_m"] = 0.0

    remaining = haz[~haz[HAZ_ID].isin(within[HAZ_ID])]
    if len(remaining):
        nearest = gpd.sjoin_nearest(
            remaining.to_crs(distance_crs),
            ops.to_crs(distance_crs),
            how="inner",
            max_distance=max_distance_m,
            distance_col="distance_m",
        )
        # keep one OP per HAZ if several are equally near
        nearest = (
            nearest.sort_values(["distance_m", OP_ID])
            .drop_duplicates(HAZ_ID)[[HAZ_ID, OP_ID, "distance_m"]]
            .assign(link="nearest")
        )
    else:
        nearest = pd.DataFrame(columns=[HAZ_ID, OP_ID, "distance_m", "link"])

    links = pd.concat([within, nearest], ignore_index=True)
    n_unlinked = haz[HAZ_ID].nunique() - links[HAZ_ID].nunique()
    logging.info(
        "Linked HAZs to OPs: %d within, %d nearest, %d not linked",
        within[HAZ_ID].nunique(),
        nearest[HAZ_ID].nunique(),
        n_unlinked,
    )
    links[HAZ_ID] = links[HAZ_ID].astype(str)
    return links[[HAZ_ID, OP_ID, "link", "distance_m"]].reset_index(drop=True)


def haz_event_rp(events: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Table of (haz_id, event.id, rp), using the geometric mean over OPs."""
    linked = events[[EVENT_ID, OP_ID, "rp"]].merge(links[[HAZ_ID, OP_ID]], on=OP_ID, how="inner")
    linked = linked[linked.rp > 0]
    linked["log_rp"] = np.log(linked.rp)
    out = (
        linked.groupby([HAZ_ID, EVENT_ID], sort=True)["log_rp"]
        .mean()
        .pipe(np.exp)
        .rename("rp")
        .reset_index()
    )
    return out
