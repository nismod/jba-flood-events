# Workflow design

Flood depths, damages and risk for transport assets under each observed (`Obs`)
or simulated (`Sim`) fluvial flood event, per hydrological accumulation zone (HAZ).

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
    admin/adm1.gpkg                                 admin-1 polygons with GID_1 and GID_0 (e.g. GADM)
    population/...                                  GHS-POP grid and urban centres, for the draft
                                                    population-disconnected metric
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

    prepared/validation.json                           validate_inputs: fails early, listing every problem
    prepared/hazard_grid.json                          hazard_grid: shared grid of the RP maps
    prepared/assets/{layer}.parquet                    prepare_assets: grid CRS, single-part,
                                                       Hilbert-sorted, bbox covering column
    prepared/events/haz_op_links.parquet               link_haz_op: haz_id, op.id, link (within|nearest), distance_m
    prepared/events/{TYPE}_haz_event_rp.parquet        haz_event_rp: haz_id, event.id, rp (geometric mean over OPs)
    exposure/{layer}/haz={HAZ}/splits.parquet          split_assets: one row per asset piece
    asset_event_damages/{TYPE}/{layer}/haz={HAZ}/part.parquet
                                                       event_damages: one row per (asset, adm1, event) with depth > 0
    event_losses/{TYPE}/{layer}/{haz,adm1,adm0}.parquet
                                                       aggregate_losses: one row per (region, event)
    risk/{layer}/{asset,haz,adm1,adm0}_expected_annual.parquet
    risk/{layer}/{asset,haz,adm1,adm0}_exceedance.parquet
    risk/all/{haz,adm1,adm0}_*.parquet                 risk_metrics: Sim set; `all` sums damage over layers
    qa/exposure_summary.csv                            qa_summary: assets, pieces and wet size per layer and HAZ
    disconnection/...                                  draft population metrics and criticality (below)

There are two jobs per HAZ. `split_assets` reads the raster windows once and
splits every layer. `event_damages` handles every layer and event set.

The `haz=` directories are Hive partitions. Each output directory reads as one
dataset with a `haz` column:

```python
splits = gpd.read_parquet("processed_data/exposure/road_edges")
damages = pd.read_parquet("processed_data/asset_event_damages/Sim/road_edges")
per_asset_event = damages.groupby(["id", "event_id"]).damage_undefended_usd.sum()
```

### splits.parquet

| column | |
|---|---|
| `split_id` | piece id, unique within the HAZ; (`haz`, `split_id`) is globally unique |
| `id`, `cost_type` | from the asset layer |
| `adm1` | admin-1 region containing the piece; null if none |
| `raster_i`, `raster_j` | global grid cell |
| `size`, `size_unit` | piece length (m), area (m2), or 1 (count) for points, in local UTM |
| `sop` | standard of protection RP, the max over defended areas containing the piece; NaN if undefended |
| `depth_rp{RP}` | mapped undefended depth for the cell at each RP |
| `geometry` | piece geometry, in the hazard grid CRS |

### part.parquet (asset event damages)

Pieces are summed to the asset within each HAZ and admin-1 region, so an asset
that crosses a boundary has one row per side. Sum over `haz` and `adm1` for
whole-asset values.

| column | |
|---|---|
| `id`, `adm1` | asset and admin-1 region |
| `event_id`, `rp` | event and its RP in this HAZ |
| `max_depth_m` | deepest interpolated undefended depth over the asset's pieces |
| `max_depth_defended_m` | the same, ignoring pieces protected in this event |
| `exposed_size`, `exposed_size_defended` | flooded length, area or count |
| `damage_undefended_usd` | sum of size × unit cost × damage fraction |
| `damage_defended_usd` | the same, with protected pieces at zero |

Damage is null for exposure-only assets (no `cost_type`), e.g. nodes.

The piece-by-event table isn't stored: it is a function of `splits.parquet`
and the event RPs, and can be very large. Rebuild it for chosen events with
`jba_events.damage.piece_event_table`.

### Risk tables

`*_expected_annual.parquet` has `expected_annual_{value}` per key.
`*_exceedance.parquet` has one row per key and `probability`, with
`return_period` (1 / probability) and a column per value. Values are
`damage_undefended_usd`, `damage_defended_usd`, `exposed_size` and
`exposed_size_defended`; the `all` tables hold damage only.

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

**Risk.** The Sim set represents `risk.years` years (10,000). Expected annual
damage is the sum of event losses divided by the years. For exceedance, the
k-th largest event loss is exceeded k times in N years, so its annual
exceedance probability is 1 − exp(−k / N). This is an occurrence probability
(largest single event in a year). An annual-aggregate version needs the
event-to-year mapping, which the workflow doesn't read yet. Probabilities are
set in `risk.exceedance_probabilities`.

## Population disconnected and detoured (draft)

Inputs, set under `disconnection` in the config: JRC GHS-POP 2020 1 km
population grid, the JRC GHSL Urban Centre Database, and a network layer
(`road_edges`). The network is built with snkit 2.0 (prerelease, from the
`claude/spatial-network-api-nbfu58` branch; the `network` extra).

**Baseline** (`disconnection_baseline`, once):

1. `snkit.from_edges` builds the road network, joining line ends within
   `node_tolerance_m`.
2. Each populated cell centre and each urban centre (its representative
   point) is added as a node and blended in with `snkit.link_nodes_to_edges`:
   projected onto its nearest road within `max_snap_m`, the road split
   there, and a connector edge added. Split pieces keep their road's asset
   id, so they fail with it; connectors have none and never fail. Snapping is
   done in `network_crs`; route lengths are geodesic.
3. Sources are the blended centre nodes and every node inside a centre
   polygon, so people living in an urban centre are already at one.
4. One multi-source shortest-path search gives each node its distance d0 to
   the nearest centre, and a shortest-path tree. Each node's subtree is
   recorded as an interval of a preorder walk of the tree.

Population cells are `served`, `too_far` (no road within `max_snap_m`) or
`no_route` (on a part of the network with no centre); `summary.json` totals them.

**Failures.** An asset fails in an event when its maximum depth reaches
`depth_threshold_m` (0.5 m). When a set of edges is removed, only nodes in
the subtrees below removed tree edges can change distance. Those are found
from the preorder intervals, and their new distances d1 come from a search
over those nodes alone: from a virtual source joined to each of them at
d0 of its intact outside neighbour plus the edge length. This is exact (tested
against full searches on random networks) and its cost follows the size of
the cut-off subtrees, not the size of the network.

A served person is

- **disconnected** if d1 is infinite: no centre can be reached
- **detoured** if d1 > d0 (by more than 1 m), adding d1 − d0 to their trip.

**Per event** (`disconnection_events`, in `chunks` parallel parts):

    disconnection/{TYPE}/events/part-{chunk}.parquet
        event_id, protection (undefended|defended), adm1, adm0,
        population_disconnected, population_detoured, detour_person_km
    disconnection/Sim/risk/{adm1,adm0,total}_expected_annual.parquet
        annual_probability_any, expected_annual_population_disconnected,
        expected_annual_population_detoured, expected_annual_detour_person_km
    disconnection/Sim/risk/{adm1,adm0,total}_exceedance.parquet

Expected annual values count people × events per year. Regions are those of
the population cell, not of the flooded road.

**Static criticality** (`network_criticality`, in `criticality_chunks`
parallel parts, needing only the baseline): each asset is failed on its own.

    disconnection/criticality.parquet
        asset_id, population_routed (whose baseline route uses the asset),
        population_disconnected, population_detoured, detour_person_km,
        annual_failure_frequency_undefended, _defended (Sim events per year in
        which the asset fails), expected_annual_population_disconnected_if_alone,
        expected_annual_detour_person_km_if_alone

Ranked by population disconnected, then detour. The "if alone" columns
multiply single-failure impact by failure frequency. They rank assets by flood
risk, but aren't the asset's share of event impacts: in an event, other assets
fail at the same time.

Known limits: distance only, with no speeds, road classes or congestion;
centres are equivalent destinations; one network layer; cells further than
`max_snap_m` beyond the network extent aren't counted.

## Running

    snakemake --profile profiles/local all     # one machine
    snakemake --profile profiles/slurm all     # SLURM
    pip install -e .                           # the jba_events package (in environment.yml)
    pytest                                     # synthetic-data tests, incl. a full Snakemake run

Data paths come from `config.json` at the repository root (copy
`config.example.json`). Use `--config paths_file=...` to use another file.
