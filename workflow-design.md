# Workflow design

Undefended fluvial flood depths and damages for each transport asset piece
and each observed (`Obs`) or simulated (`Sim`) event, per hydrological
accumulation zone (HAZ).

## Principles

- **Read subsets, don't clip.** Jobs read a window of the continental return
  period (RP) maps and a bbox subset of each prepared continental asset layer.
  Nothing is copied per HAZ.
- **One global grid.** All RP maps must share one grid (checked by
  `hazard_grid`). `raster_i` (column) and `raster_j` (row) are always global
  cell indices, so tables from different HAZs can be joined or concatenated.
- **Cells belong to exactly one HAZ**, the one containing the cell centre.
  An asset piece belongs to the HAZ of its cell, so there's no loss or double
  counting at HAZ boundaries.
- **Shared work is done once.** OP linking and (HAZ, event) RPs are computed
  once per event set, not in every per-HAZ job.
- **Fail loudly.** A missing input layer, cost table or damage curve for a
  costed asset is an error, not an empty output. Legitimately empty results
  (no assets or no events in a HAZ) are written as empty tables with the
  full schema.

## Inputs

Paths are relative to `paths.data` and set in `workflow/config.yaml`.

    hazards/flrf_ud_Q{RP}.tif                       RP = 20, 50, 100, 200, 500, 1500
    basins/haz_500.gpkg                             HAZ polygons, id column T500_ID
    events/RiverOpInfo.gpkg                         river OPs (op.id)
    events/{TYPE}EventRp.csv                        event.id, op.id, rp; TYPE = Obs, Sim
    defended_areas/defended_areas.gpkg              defended polygons, JBA_SoP
    infrastructure/africa_{layer}_network.parquet   one file per layer, e.g. road_edges,
                                                    railway_nodes, airport_field_polygons;
                                                    must have a unique `id` and should have `cost_type`
    costs/{layer}.csv                               cost_type, mean_cost_usd_per_m | _per_sqm | mean_cost_usd
    curves/{asset_type}/{cost_type}.csv             flood_depth_m, damage_fraction_mean

The layer suffix (`_edges`, `_nodes`, `_polygons`) sets the geometry kind
(line, point, polygon) and the size unit (m, m2, count). By default the layers
are every source file that matches; set `assets.layers` to choose.

The preprocessing scripts (`workflow/preprocess.py`, `roads_preprocess.py`,
`airports_preprocess.py`, `preprocess_cost_type.py`) produce these from the
incoming data and are run by hand before the Snakefile.

## Rules and outputs

    prepared/hazard_grid.json                          hazard_grid: shared grid of the RP maps
    prepared/assets/{layer}.parquet                    prepare_assets: grid CRS, single-part,
                                                       Hilbert-sorted, bbox covering column
    prepared/events/haz_op_links.parquet               link_haz_op: haz_id, op.id, link (within|nearest), distance_m
    prepared/events/{TYPE}_haz_event_rp.parquet        haz_event_rp: haz_id, event.id, rp (geometric mean over OPs)
    exposure/{layer}/haz={HAZ}/splits.parquet          split_assets: one row per asset piece
    event_damages/{TYPE}/{layer}/haz={HAZ}/part.parquet
                                                       event_damages: one row per (piece, event) with depth > 0

The `haz=` directories are Hive partitions. Each output directory reads as one
dataset with a `haz` column:

```python
splits = gpd.read_parquet("processed_data/exposure/road_edges")
damages = pd.read_parquet("processed_data/event_damages/Sim/road_edges")
per_asset_event = damages.groupby(["id", "event_id"]).damage_undefended_usd.sum()
```

### splits.parquet

| column | |
|---|---|
| `split_id` | piece id, unique within the HAZ; (`haz`, `split_id`) is globally unique |
| `id`, `cost_type` | from the asset layer |
| `raster_i`, `raster_j` | global grid cell |
| `size`, `size_unit` | piece length (m), area (m2), or 1 (count) for points, in local UTM |
| `sop` | standard of protection RP, the max over defended areas containing the piece; NaN if undefended |
| `depth_rp{RP}` | mapped undefended depth for the cell at each RP |
| `geometry` | piece geometry, in the hazard grid CRS |

### part.parquet (event damages)

| column | |
|---|---|
| `split_id`, `id` | piece and asset |
| `event_id`, `rp` | event and its RP in this HAZ |
| `depth_m` | interpolated undefended depth |
| `protected` | event RP ≤ piece SoP |
| `damage_fraction` | from the damage curve for the piece's cost_type |
| `damage_undefended_usd` | size × unit cost × damage fraction |
| `damage_defended_usd` | 0 if protected, else the undefended damage |

Damage is null for exposure-only pieces (no `cost_type`), e.g. nodes.

## Method

**HAZ to OP.** OPs inside a HAZ are linked to it. A HAZ with none inside
takes the OP nearest its boundary within `events.nearest_op_max_distance_m`
(100 km), measured in `events.distance_crs`. Otherwise the HAZ isn't at risk.

**Event RP per HAZ.** Geometric mean of the RPs of the HAZ's OPs in the event.

**Depth.** Per cell, with mapped RPs r_1..r_n and depths d_1..d_n, let r_f be
the first RP with depth > 0. The zero-depth RP z is the HORP
(`hazard.horp`, 11) if r_f = r_1, else the mean of r_f and r_{f-1}. Depth is 0
for rp ≤ z, log-linear from (z, 0) to (r_f, d_f), log-linear between later
mapped RPs, and d_n for rp ≥ r_n.

**Defended.** Depth (and damage) is zero when the event RP ≤ the piece's SoP,
otherwise it's the undefended value.

## Running

    snakemake --profile profiles/local all     # one machine
    snakemake --profile profiles/slurm all     # SLURM
    pytest                                     # synthetic-data tests, incl. a full Snakemake run

Data paths come from `config.json` at the repository root (copy
`config.example.json`). Use `--config paths_file=...` to use another file.
