#!/usr/bin/env python3
"""
Create a country raster on the same grid as a MODIS land-cover raster.

Rules:
- land-cover value == 0  -> output 0   (water)
- land-cover value != 0  -> output country ID based on country polygons

Inputs are set in the parameter section near the top of this file.
No command line interface is used.

Outputs:
1. A GeoTIFF country raster on the MODIS grid
2. A CSV legend mapping country_id to country name / ISO code
"""

from pathlib import Path
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import os

# SOME PARAMETERS
LANDCOVER_FILE = "../output/landcover/UMD/landcover_2001.tif"
COUNTRIES_FILE = "../data_supporting/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"

OUTPUT_COUNTRY_RASTER = "../output/country_map/modis_country_map.tif"
OUTPUT_COUNTRY_LEGEND = "../output/country_map/modis_country_legend.csv"

LANDCOVER_BAND = 1
WATER_VALUE = 0
ALL_TOUCHED = False
ASSIGN_UNMATCHED_LAND_TO_NEAREST_COUNTRY = True

# Optional:
# If True, land pixels not covered by any country polygon are set to -1.
# If False, they remain 0, the same as water.
MARK_LAND_WITHOUT_COUNTRY_AS_MINUS1 = False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def pick_country_fields(gdf: gpd.GeoDataFrame) -> Tuple[str, str]:
    """Choose sensible country name and ISO code columns."""
    possible_name_fields = [
        "NAME", "ADMIN", "SOVEREIGNT", "COUNTRY", "NAME_EN", "NAME_LONG"
    ]
    possible_iso_fields = [
        "ISO_A3", "ADM0_A3", "GU_A3", "SOV_A3"
    ]

    name_field = next((c for c in possible_name_fields if c in gdf.columns), None)
    iso_field = next((c for c in possible_iso_fields if c in gdf.columns), None)

    if name_field is None:
        raise ValueError(
            "Could not find a country name field in the country file.\n"
            f"Available columns: {list(gdf.columns)}"
        )

    if iso_field is None:
        raise ValueError(
            "Could not find an ISO A3 field in the country file.\n"
            f"Available columns: {list(gdf.columns)}"
        )

    return name_field, iso_field


def read_countries_compat(countries_path: Path) -> gpd.GeoDataFrame:
    """Read country polygons with a fallback for older Fiona/GeoPandas combos."""
    try:
        return gpd.read_file(str(countries_path))
    except AttributeError as exc:
        if "fiona" not in str(exc) or "path" not in str(exc):
            raise

        import fiona

        with fiona.open(str(countries_path)) as src:
            features = list(src)
            crs = src.crs_wkt if src.crs_wkt else src.crs

        return gpd.GeoDataFrame.from_features(features, crs=crs)


def _shift_with_zero_padding(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift array by dy/dx with zeros padded at exposed edges."""
    out = np.zeros_like(arr)

    src_row_start = max(0, -dy)
    src_row_end = arr.shape[0] - max(0, dy)
    src_col_start = max(0, -dx)
    src_col_end = arr.shape[1] - max(0, dx)

    dst_row_start = max(0, dy)
    dst_row_end = arr.shape[0] - max(0, -dy)
    dst_col_start = max(0, dx)
    dst_col_end = arr.shape[1] - max(0, -dx)

    out[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = arr[
        src_row_start:src_row_end, src_col_start:src_col_end
    ]
    return out


def assign_unmatched_land_to_nearest_country(
    country_raster: np.ndarray,
    non_water_mask: np.ndarray,
) -> np.ndarray:
    """Fill land cells with country_id 0 using nearest existing country IDs."""
    filled = country_raster.copy()
    unmatched_land = non_water_mask & (filled == 0)

    if not np.any(unmatched_land):
        return filled

    if not np.any(filled > 0):
        raise ValueError("No country pixels were rasterized; cannot assign nearest country.")

    neighbor_offsets = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    while True:
        previous_unmatched_count = int(unmatched_land.sum())
        candidate_ids = np.zeros_like(filled)

        for dy, dx in neighbor_offsets:
            shifted = _shift_with_zero_padding(filled, dy, dx)
            can_fill = unmatched_land & (candidate_ids == 0) & (shifted > 0)
            candidate_ids[can_fill] = shifted[can_fill]

        updated = candidate_ids > 0
        if not np.any(updated):
            break

        filled[updated] = candidate_ids[updated]
        unmatched_land = non_water_mask & (filled == 0)

        if int(unmatched_land.sum()) == previous_unmatched_count:
            break

    return filled


def main() -> None:
    landcover_path = Path(LANDCOVER_FILE)
    countries_path = Path(COUNTRIES_FILE)
    output_raster_path = Path(OUTPUT_COUNTRY_RASTER)
    output_legend_path = Path(OUTPUT_COUNTRY_LEGEND)

    # make the output directory if it doesn't exist
    os.makedirs(output_raster_path.parent, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Read land-cover raster to get grid information
    # -------------------------------------------------------------------------
    with rasterio.open(landcover_path) as src:
        landcover = src.read(LANDCOVER_BAND)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width

    if crs is None:
        raise ValueError("The land-cover raster has no CRS.")

    non_water_mask = landcover != WATER_VALUE

    # -------------------------------------------------------------------------
    # 2. Read country polygons
    # -------------------------------------------------------------------------
    countries = read_countries_compat(countries_path)

    if countries.empty:
        raise ValueError("The country polygon file contains no features.")

    if countries.crs is None:
        raise ValueError("The country polygon file has no CRS.")

    name_field, iso_field = pick_country_fields(countries)

    # Keep valid geometries only
    countries = countries[countries.geometry.notnull()].copy()
    countries = countries[~countries.geometry.is_empty].copy()

    # Reproject to raster CRS
    countries = countries.to_crs(crs)

    # Assign integer IDs
    countries = countries.reset_index(drop=True)
    countries["country_id"] = np.arange(1, len(countries) + 1, dtype=np.int32)

    # -------------------------------------------------------------------------
    # 3. Rasterize country polygons onto the MODIS grid
    # -------------------------------------------------------------------------
    shapes = ((geom, cid) for geom, cid in zip(countries.geometry, countries["country_id"]))

    country_raster = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=ALL_TOUCHED,
    )

    # -------------------------------------------------------------------------
    # 4. Set water cells to 0
    # -------------------------------------------------------------------------
    country_raster[~non_water_mask] = 0

    if ASSIGN_UNMATCHED_LAND_TO_NEAREST_COUNTRY:
        country_raster = assign_unmatched_land_to_nearest_country(country_raster, non_water_mask)

    # Optional: distinguish "land but no country" from water
    if MARK_LAND_WITHOUT_COUNTRY_AS_MINUS1:
        mask = non_water_mask & (country_raster == 0)
        country_raster[mask] = -1

    

    # -------------------------------------------------------------------------
    # 5. Write output raster
    # -------------------------------------------------------------------------
    profile.update(
        dtype="int32",
        count=1,
        compress="lzw",
        nodata=0 if not MARK_LAND_WITHOUT_COUNTRY_AS_MINUS1 else None,
    )

    output_raster_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_raster_path, "w", **profile) as dst:
        dst.write(country_raster, 1)

    # -------------------------------------------------------------------------
    # 6. Write legend CSV
    # -------------------------------------------------------------------------
    legend = countries[["country_id", name_field, iso_field]].copy()
    legend.columns = ["country_id", "country_name", "iso_a3"]

    if MARK_LAND_WITHOUT_COUNTRY_AS_MINUS1:
        extra = pd.DataFrame(
            [{"country_id": -1, "country_name": "Land without country", "iso_a3": ""}]
        )
        legend = pd.concat([extra, legend], ignore_index=True)

    water_row = pd.DataFrame(
        [{"country_id": 0, "country_name": "Water", "iso_a3": ""}]
    )
    legend = pd.concat([water_row, legend], ignore_index=True)

    output_legend_path.parent.mkdir(parents=True, exist_ok=True)
    legend.to_csv(output_legend_path, index=False)

    print(f"Wrote raster: {output_raster_path}")
    print(f"Wrote legend: {output_legend_path}")


if __name__ == "__main__":
    main()