"""Split every asset layer on the hazard grid within one HAZ.

The script reads a window of each continental RP map once, then a bbox subset
of each prepared asset layer. It writes one piece table per layer; see
`jba_events.exposure.split_layer` for the columns.

Example:

    python workflow/scripts/split_assets.py \
        --haz_id 500_13_19495 --haz_path processed_data/basins/haz_500.gpkg --haz_id_column T500_ID \
        --grid processed_data/prepared/hazard_grid.json \
        --rp 20 --rp_path processed_data/hazards/flrf_ud_Q20.tif ... \
        --layer road_edges --layer railway_edges \
        --asset_template 'processed_data/prepared/assets/{layer}.parquet' \
        --defended_path processed_data/defended_areas/defended_areas.gpkg --sop_column JBA_SoP \
        --admin_path processed_data/admin/adm1.gpkg --adm1_column GID_1 \
        --output_template 'processed_data/exposure/{layer}/haz=500_13_19495/splits.parquet'
"""

import logging
import os

import click

from jba_events.exposure import HazWindow, read_haz, split_layer
from jba_events.grid import HazardGrid


@click.command()
@click.option("--haz_id", required=True)
@click.option("--haz_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--haz_id_column", required=True)
@click.option("--grid", "grid_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--rp", "rps", required=True, multiple=True, type=float, help="Mapped RP, repeat in order")
@click.option("--rp_path", "rp_paths", required=True, multiple=True, type=click.Path(exists=True), help="RP map for each --rp")
@click.option("--layer", "layers", required=True, multiple=True)
@click.option("--asset_template", required=True, help="Prepared asset path, with {layer}")
@click.option("--defended_path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--sop_column", default="JBA_SoP", show_default=True)
@click.option("--admin_path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--adm1_column", default=None)
@click.option("--output_template", required=True, help="Output path, with {layer}")
def main(haz_id, haz_path, haz_id_column, grid_path, rps, rp_paths, layers, asset_template,
         defended_path, sop_column, admin_path, adm1_column, output_template):
    if len(rps) != len(rp_paths) or list(rps) != sorted(rps):
        raise ValueError("Give one --rp_path per --rp, with --rp in increasing order")
    grid = HazardGrid.from_json(grid_path)
    hw = HazWindow(read_haz(haz_path, haz_id_column, haz_id), grid, rps, rp_paths)
    logging.info("HAZ %s window: %s", haz_id, hw.window)

    on_grid = not hw.window.is_empty
    defended = hw.read_bbox(defended_path, [sop_column]) if defended_path and on_grid else None
    admin = hw.read_bbox(admin_path, [adm1_column]) if admin_path and on_grid else None

    for layer in layers:
        pieces = split_layer(hw, layer, asset_template.format(layer=layer), defended, sop_column, admin, adm1_column)
        output = output_template.format(layer=layer)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        pieces.to_parquet(output, index=False)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(filename)s %(message)s", level=logging.INFO)
    main()
