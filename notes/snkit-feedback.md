# snkit 2.0 prerelease: notes from the jba-events integration

Notes from using snkit at `claude/spatial-network-api-nbfu58` (2.0.0a1) to
blend population cells and urban centres into a road network, for the
population disconnected and detour metrics in this repository
(`src/jba_events/disconnection.py`).

## What worked well

- `link_nodes_to_edges(how="nearest", max_distance=..., nodes=mask,
  tolerance=...)` does exactly the blending needed: split at the projection,
  add a connector, and attach to an existing end within tolerance rather than
  splitting next to it.
- Split pieces carry their parent's attributes, so a road's asset id follows
  every piece and the pieces fail together.
- A string `id` column is kept as `id_original`, not used for topology: no
  surprises about ids.
- Speed: 2M road edges plus 2M points blended in about 22 s on the array path
  (12 s for the link itself), and 1.4M roads plus 1M cells in 22 s at a
  4.6 GB peak.

## Rough edges met

1. **No public way to add nodes to an existing network.** Adding population
   points to a road network built with `from_edges(tolerance=...)` took
   `from_topology` with a hand-built node table. `from_nodes_and_edges` would
   do it in one step, but joins line ends to points rather than to each other,
   so it can't also join near road ends. A public `add_nodes(network,
   points)` (the edit already exists internally) would make this one call.
2. **Silent renumbering in `from_topology`.** Concatenating uint64 snkit ids
   with int64 ids gives float64, and `from_topology` then treats them as
   non-integer, renumbers everything and adds `id_original`,
   `from_id_original` and `to_id_original`. A warning, or accepting
   integer-valued floats, would catch this.
3. **Sparse ids.** After `from_edges(tolerance=...)` merges ends, node ids
   have gaps. Every graph library needs dense 0..n-1 indices, so each user
   writes the same `searchsorted` mapping. An option for dense ids, or the
   export in point 5, would avoid that.
4. **Connectors aren't marked.** New connector edges are told apart only by
   having null attributes. A column naming the node each connector serves
   (or a flag) would be more robust, e.g. `link_nodes_to_edges(...,
   link_column="linked_node")`.
5. **No graph export for the new types.** `to_networkx` and `to_igraph` live
   in the legacy module and work on GeoDataFrames. For `ArrayNetwork`, an
   export of `(from_index, to_index, length)` arrays with dense indices
   (`to_csgraph`, or a scipy CSR) would let routing start without touching
   geometry.
6. **`ArrayNetwork.to_crs` changes the network in place**, unlike the
   documented convention that operations are pure.
7. **Projected CRS needed for snapping distances.** At continental scale one
   equidistant projection distorts snapping distances by a few percent, and
   lengths still have to be computed geodesically afterwards. Geodesic
   tolerance in the kernel for geographic CRSs (haversine or Vincenty for
   point-to-segment distance), and a geodesic `edge_lengths(network)`, would
   remove both steps.
8. **The `merge_nodes` default would lose attributes here.** Its rule keeps an
   attribute only where all members agree, so a population node merged with a
   road node (population null) would lose its population. That default is
   right in general, but worth a note in the docstring with `agg={...: "sum"}`
   as the fix.

## Performance

- **Memory from GeoDataFrames.** `from_nodes_and_edges` on 2M lines and 2M
  points peaked at 3.3 GB, and the link at 5.7 GB: several times the ~190
  bytes per edge of the kernel. Most of it is the geometry objects coming in
  and string attribute columns copied per piece. Reading straight from
  GeoParquet (`snkit.io.read_parquet`) and keeping string attributes as
  dictionary-encoded (categorical) columns through splits should cut this.
- **Out-of-core linking.** Linking is the operation this workflow needs at
  full scale (all GHS-POP populated cells in Africa, roughly 10M or more
  points), and is listed as not yet done out of core. Tiles that overlap by
  `max_distance` would fit the existing build pattern.

## Possible additions informed by this work

- **Routing on kernel arrays.** Multi-source shortest paths and the
  shortest-path tree are the first step of any accessibility or criticality
  analysis. In the Rust kernel they would be available to R and JavaScript
  too, and avoid building a scipy matrix (which also adds parallel edges
  together instead of keeping the shortest: `jba_events.disconnection.min_csr`
  works around that).
- **Incremental re-routing after failures.** The method in
  `RouteNetwork.reroute` is general: record preorder intervals of the
  shortest-path tree once, then for any set of removed edges recompute only
  the cut-off subtrees, from a virtual source joined to their intact
  neighbours at baseline distance. It is exact, and costs milliseconds per
  failure set on a 3M-edge network instead of a full search. A kernel
  `reroute(failed_edges)` would make event-based and static criticality
  analysis cheap for every snkit user.
- **Bridges and 2-edge-connected components** in the kernel, alongside
  `label_components`, for a fast first screen of which edges can disconnect
  anything at all.
