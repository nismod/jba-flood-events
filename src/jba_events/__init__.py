"""Flood event exposure, damage and risk for transport assets from JBA event sets.

Modules:

- grid: the continental hazard grid and per-HAZ windows of it
- split: split asset geometries on a grid window
- exposure: asset pieces within one HAZ, with mapped depths, SoP and admin region
- events: link observation points to HAZs, return period per (HAZ, event)
- depth: interpolate event depth from mapped return period depths
- damage: unit costs, damage curves, and piece or asset tables per event
- risk: expected annual values and exceedance values from an event set
- disconnection: draft population-disconnected metric

The command-line scripts in `workflow/scripts/` and the `Snakefile` use this package.
"""
