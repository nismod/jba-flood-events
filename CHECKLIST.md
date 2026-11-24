# Before running on real data

The workflow has only been run on the synthetic data in `tests/`. These are the
things to check or set before trusting results from the JBA datasets. Settings
are in `workflow/config.yaml` unless noted.

## Set

- [ ] **Data paths.** Copy `config.example.json` to `config.json` and set
      `paths.data`.
- [ ] **Admin boundaries** (`admin`). The defaults are guesses for a GADM-style
      file: `admin/adm1.gpkg` with `GID_1` and `GID_0`. Set `path`,
      `adm1_column` and `adm0_column` to match your file.
- [ ] **Population grid** (`disconnection.population_path`). The default is the
      GHS-POP R2023A, 2020, Mollweide 1 km filename under `population/`.
- [ ] **Urban centres** (`disconnection.centres_path`, `centre_id_column`). The
      default is `population/GHS_UCDB.gpkg` with `ID_UC_G0`. Polygons are
      best; points also work.
- [ ] **Years in the simulated set** (`risk.years`, default 10,000). Every
      expected annual value and exceedance probability scales with this.
- [ ] **Exceedance probabilities** (`risk.exceedance_probabilities`).
- [ ] **Asset layers** (`assets.layers`). Empty means every
      `infrastructure/africa_{layer}_network.parquet` ending in `_edges`,
      `_nodes` or `_polygons`. Each needs a unique `id` column.
- [ ] **SLURM** (`profiles/slurm/config.yaml`): partition and account. The
      profile hasn't been run on a cluster.

## Check

- [ ] **Event IDs are unique across regions.** `workflow/preprocess.py`
      concatenates the regional `SimEventRp_*.csv` files. If regions reuse
      event IDs, different events would be averaged as one. `validate_inputs`
      catches duplicate (event, OP) rows, but not reuse with different OPs.
- [ ] **Exceedance basis.** With no event-to-year mapping, probabilities
      describe the largest single event in a year, not a year's summed losses.
      Expected annual values aren't affected. If `SimEventInfo.csv` has a year
      column, an annual-aggregate version can be added.
- [ ] **Road network is noded at edge ends.** The population-disconnected
      metric rebuilds topology from edge endpoints. Check
      `disconnection/baseline/summary.json`: a large `no_route` share means
      the network is fragmented, and a large `too_far` share means the snap
      distance (`disconnection.max_snap_m`) is too small.
- [ ] **Flood threshold for road failure** (`disconnection.depth_threshold_m`,
      default 0.5 m).
- [ ] **Exposure totals.** Compare `qa/exposure_summary.csv` against the
      source layers, and look for HAZs with unexpectedly little exposure.
- [ ] **HAZ to OP links.** In `prepared/events/haz_op_links.parquet`, check
      how many HAZs are linked by `nearest` and how many aren't linked at all
      (no OP within `events.nearest_op_max_distance_m`).
- [ ] **Output size.** `asset_event_damages/Sim/` is the largest output. If it
      is too large, a minimum-depth filter is the simplest reduction.
- [ ] **README equation.** It still shows depth = 0 for r ≤ 2. The workflow
      uses the HORP (11) and the mean-of-bracketing-RPs rule described beside it.

## Not done yet

- [ ] **CI.** A GitHub Actions workflow couldn't be pushed from the session
      that wrote this. It only needs `pip install -e ".[dev]"` then `pytest`.
- [ ] **Preprocessing in the Snakefile.** `workflow/preprocess.py`,
      `roads_preprocess.py`, `airports_preprocess.py` and
      `preprocess_cost_type.py` (which overwrites files in place) still run by
      hand first.
- [ ] **Uncertainty.** Only mean unit costs and mean damage fractions are used.
- [ ] **Real-data test.** A small real extract (one HAZ) as a test fixture,
      ideally compared against the earlier Jamaica scripts.
