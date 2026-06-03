'''
User manual:
This script extracts maps of land cover, NPP, LAI, and FPAR for a specified country
from MODIS data and saves the output as GeoTIFF files and CSV spreadsheets.

Memory-safe rewrite:
  - Does NOT allocate full global MODIS arrays.
  - Does NOT mask full global maps with np.where.
  - Reads only the MODIS tiles/windows overlapping the selected country.
  - Reads only cropped latitude/longitude windows.

Usage:
  python extract_country_maps_memory_safe.py --country <CountryName> --yearstart <StartYear> --yearend <EndYear> [--onjupyter]

Example:
  python extract_country_maps_memory_safe.py --country Kenya --yearstart 2003 --yearend 2010
'''

import argparse
import csv
import gc
import os
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from pyhdf.SD import SD, SDC
from rasterio.windows import Window


# --------------------------------------------------------------------------------
# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract country-level MODIS land cover, NPP, LAI, and FPAR maps."
    )
    parser.add_argument("--country", type=str, required=True, help="Country name")
    parser.add_argument("--yearstart", type=int, required=True, help="Start year")
    parser.add_argument("--yearend", type=int, required=True, help="End year")
    parser.add_argument("--onjupyter", action="store_true", help="Use Jupyter/P-drive paths")

    args = parser.parse_args()
    if args.yearstart > args.yearend:
        parser.error("yearstart must be less than or equal to yearend")
    return args


args = parse_args()

print("Parameters received:")
print(f"Country   : {args.country}")
print(f"Year start: {args.yearstart}")
print(f"Year end  : {args.yearend}")
print(f"OnJupyter : {args.onjupyter}")

country = args.country
yearstart = args.yearstart
yearend = args.yearend

if args.onjupyter:
    main_dir = "/pdrive/projects/pflam/projects/Global/data/modis/"
else:
    main_dir = "../"


# --------------------------------------------------------------------------------
# MODIS grid constants

res = 463.3127165275005  # meters
ntiles_x = 36
ntiles_y = 18
npixels_per_tile_x = 2400
npixels_per_tile_y = 2400

ORIGIN_X = -20015109.354
ORIGIN_Y = 10007554.677

COUNTRY_RASTER = f"{main_dir}output/country_map/modis_country_map.tif"
COUNTRY_LEGEND = f"{main_dir}output/country_map/modis_country_legend.csv"

input_directory_landcover = f"{main_dir}data/LandCover/"
input_directory_npp = f"{main_dir}data/NPP/"
input_directory_lai = f"{main_dir}data/LAI_fpar/"
input_directory_fpar = f"{main_dir}data/LAI_fpar/"
output_directory_main = f"{main_dir}output/country_maps/"


# --------------------------------------------------------------------------------
# Helpers

def _print(msg):
    """Print a message with a timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")


def get_quantity_directory(quantity, year=None):
    """Return the input directory for a MODIS quantity."""
    if quantity == "LandCover":
        return input_directory_landcover
    if quantity == "NPP":
        return os.path.join(input_directory_npp, f"{year}/")
    if quantity == "LAI":
        return os.path.join(input_directory_lai, f"{year}/")
    if quantity == "FPAR":
        return os.path.join(input_directory_fpar, f"{year}/")
    raise ValueError(f"Unknown quantity '{quantity}'")


def get_mosaic_files(year, startday, quantity):
    """Get all HDF files for a given year, start day, and quantity."""
    directory = get_quantity_directory(quantity, year)

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Input directory does not exist: {directory}")

    startday_str = f"{startday:03d}"

    if quantity == "LandCover":
        start_filename = f"MCD12Q1.A{year}{startday_str}"
    elif quantity == "NPP":
        start_filename = f"MOD17A2HGF.A{year}{startday_str}"
    elif quantity in ("LAI", "FPAR"):
        start_filename = f"MCD15A2H.A{year}{startday_str}"
    else:
        raise ValueError(f"Unknown quantity '{quantity}'")

    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.startswith(start_filename) and f.endswith(".hdf")
    )


def parse_tile_from_filename(filename):
    """Extract MODIS horizontal/vertical tile indices from a filename."""
    # Example basename part: MOD17A2HGF.A2003001.h21v09.061....hdf
    tile_code = os.path.basename(filename).split(".")[2]
    if not (tile_code.startswith("h") and "v" in tile_code):
        raise ValueError(f"Could not parse tile code from filename: {filename}")
    itile_x = int(tile_code[1:3])
    itile_y = int(tile_code[4:6])
    return itile_x, itile_y


def get_files_by_tile(year, startday, quantity):
    """Return a dictionary mapping (h, v) tile indices to HDF filenames."""
    files_by_tile = {}
    for filename in get_mosaic_files(year, startday, quantity):
        tile = parse_tile_from_filename(filename)
        files_by_tile[tile] = filename
    return files_by_tile


def get_dataset_name(quantity):
    """Return the HDF dataset name for a MODIS quantity."""
    if quantity == "LandCover":
        return "LC_Type1"
    if quantity == "NPP":
        return "Gpp_500m"
    if quantity == "LAI":
        return "Lai_500m"
    if quantity == "FPAR":
        return "Fpar_500m"
    raise ValueError(f"Unknown quantity '{quantity}'")


def read_data_from_hdf(hdf_path, quantity):
    """Extract and process one 2400 x 2400 MODIS tile from an HDF4 file."""
    dataset_name = get_dataset_name(quantity)

    sd = SD(hdf_path, SDC.READ)
    dataset = sd.select(dataset_name)
    try:
        attrs = dataset.attributes()
        scale = attrs.get("scale_factor", 0.1)
        fill = attrs.get("_FillValue", -3000)
        data = dataset[:].astype(np.float32, copy=False)
    finally:
        # Release HDF handles as early as possible.
        try:
            dataset.endaccess()
        except Exception:
            pass
        try:
            sd.end()
        except Exception:
            pass

    if quantity == "LandCover":
        # Keep class values as float32 so NaN can be used outside the country mask.
        return data

    if quantity == "NPP":
        data[data == fill] = np.nan
        data *= scale
        data[data > 3.0] = np.nan
        data *= 456.6
        data *= 0.47
        return data

    if quantity == "LAI":
        data[data > 100] = np.nan
        data *= 0.1
        return data

    if quantity == "FPAR":
        data[data > 100] = np.nan
        data *= 0.01
        return data

    raise ValueError(f"Unknown quantity '{quantity}'")


def get_country_id():
    """Look up the integer country ID in the country legend CSV."""
    with open(COUNTRY_LEGEND, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[1] == country:
                return int(row[0])
    raise ValueError(f"Country '{country}' not found in {COUNTRY_LEGEND}")


def scan_country_extent_and_tiles(country_id):
    """
    Find the country bounding box and required MODIS tiles without loading the full
    global country raster into memory.

    Returns
    -------
    required_tiles : list[tuple[int, int]]
        MODIS (h, v) tiles containing at least one pixel of the country.
    bbox : tuple[int, int, int, int]
        (row_start, row_stop, col_start, col_stop) in global MODIS pixel coordinates.
    profile : dict
        Rasterio profile from the country raster.
    """
    required_tiles = []
    row_min = None
    row_max = None
    col_min = None
    col_max = None

    with rasterio.open(COUNTRY_RASTER) as src:
        profile = src.profile.copy()
        raster_height = src.height
        raster_width = src.width

        for itile_y in range(ntiles_y):
            for itile_x in range(ntiles_x):
                row0 = itile_y * npixels_per_tile_y
                col0 = itile_x * npixels_per_tile_x

                if row0 >= raster_height or col0 >= raster_width:
                    continue

                height = min(npixels_per_tile_y, raster_height - row0)
                width = min(npixels_per_tile_x, raster_width - col0)
                window = Window(col0, row0, width, height)

                tile_country_ids = src.read(1, window=window)
                tile_mask = tile_country_ids == country_id

                if not np.any(tile_mask):
                    del tile_country_ids, tile_mask
                    continue

                required_tiles.append((itile_x, itile_y))

                local_rows, local_cols = np.where(tile_mask)
                abs_row_min = row0 + int(local_rows.min())
                abs_row_max = row0 + int(local_rows.max())
                abs_col_min = col0 + int(local_cols.min())
                abs_col_max = col0 + int(local_cols.max())

                row_min = abs_row_min if row_min is None else min(row_min, abs_row_min)
                row_max = abs_row_max if row_max is None else max(row_max, abs_row_max)
                col_min = abs_col_min if col_min is None else min(col_min, abs_col_min)
                col_max = abs_col_max if col_max is None else max(col_max, abs_col_max)

                del tile_country_ids, tile_mask, local_rows, local_cols

    if row_min is None:
        raise ValueError(f"Country '{country}' has no pixels in {COUNTRY_RASTER}")

    # Stop indices are exclusive.
    bbox = (row_min, row_max + 1, col_min, col_max + 1)
    return required_tiles, bbox, profile


def read_country_mask_crop(country_id, bbox):
    """Read only the cropped country mask window."""
    row_start, row_stop, col_start, col_stop = bbox
    window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)

    with rasterio.open(COUNTRY_RASTER) as src:
        country_ids_crop = src.read(1, window=window)

    country_mask_crop = country_ids_crop == country_id
    del country_ids_crop
    return country_mask_crop


def output_transform_for_bbox(profile, bbox):
    """Return the affine transform for a cropped bbox."""
    row_start, _, col_start, _ = bbox
    return profile["transform"] * rasterio.Affine.translation(col_start, row_start)


def construct_country_crop_map(year, startday, required_tiles, bbox, country_mask_crop, quantity):
    """
    Construct only the cropped country-sized map, not the full global map.

    The returned array has the country bounding-box shape. Pixels outside the country
    are set to NaN.
    """
    row_start, row_stop, col_start, col_stop = bbox
    out_height = row_stop - row_start
    out_width = col_stop - col_start

    country_map = np.full((out_height, out_width), np.nan, dtype=np.float32)
    files_by_tile = get_files_by_tile(year, startday, quantity)

    missing_tiles = []

    for itile_x, itile_y in required_tiles:
        filename = files_by_tile.get((itile_x, itile_y))
        if filename is None:
            missing_tiles.append((itile_x, itile_y))
            continue

        tile_row_start = itile_y * npixels_per_tile_y
        tile_row_stop = tile_row_start + npixels_per_tile_y
        tile_col_start = itile_x * npixels_per_tile_x
        tile_col_stop = tile_col_start + npixels_per_tile_x

        # Overlap between this tile and the country bbox in global coordinates.
        overlap_row_start = max(row_start, tile_row_start)
        overlap_row_stop = min(row_stop, tile_row_stop)
        overlap_col_start = max(col_start, tile_col_start)
        overlap_col_stop = min(col_stop, tile_col_stop)

        if overlap_row_start >= overlap_row_stop or overlap_col_start >= overlap_col_stop:
            continue

        data_tile = read_data_from_hdf(filename, quantity)

        tile_rows = slice(overlap_row_start - tile_row_start, overlap_row_stop - tile_row_start)
        tile_cols = slice(overlap_col_start - tile_col_start, overlap_col_stop - tile_col_start)
        out_rows = slice(overlap_row_start - row_start, overlap_row_stop - row_start)
        out_cols = slice(overlap_col_start - col_start, overlap_col_stop - col_start)

        country_map[out_rows, out_cols] = data_tile[tile_rows, tile_cols]

        del data_tile
        gc.collect()

    if missing_tiles:
        _print(f"Warning: {quantity} missing {len(missing_tiles)} required tile(s): {missing_tiles}")

    # Mask in place to avoid allocating another country-sized array.
    country_map[~country_mask_crop] = np.nan
    return country_map


def read_lat_lon_crops(bbox, country_mask_crop):
    """Read only cropped latitude and longitude windows, then mask outside country."""
    row_start, row_stop, col_start, col_stop = bbox
    window = Window(col_start, row_start, col_stop - col_start, row_stop - row_start)

    lat_filename = f"{main_dir}data_supporting/modis500_latlon/lat_modis500.tif"
    lon_filename = f"{main_dir}data_supporting/modis500_latlon/lon_modis500.tif"

    with rasterio.open(lat_filename) as src:
        lat_grid_cropped = src.read(1, window=window).astype(np.float32, copy=False)
    lat_grid_cropped[~country_mask_crop] = np.nan

    with rasterio.open(lon_filename) as src:
        lon_grid_cropped = src.read(1, window=window).astype(np.float32, copy=False)
    lon_grid_cropped[~country_mask_crop] = np.nan

    return lat_grid_cropped, lon_grid_cropped


def write_geotiff(output_path, array, profile, transform):
    """Write a single-band GeoTIFF for a cropped country map."""
    output_profile = profile.copy()
    output_profile.update(
        count=1,
        height=array.shape[0],
        width=array.shape[1],
        transform=transform,
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
    )

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(array.astype(np.float32, copy=False), 1)


def write_country_csv(
    output_csv_path,
    lat_grid_cropped,
    lon_grid_cropped,
    landcover_cropped,
    npp_cropped,
    lai_cropped,
    fpar_cropped,
):
    """Write only valid country pixels to CSV."""
    lat = lat_grid_cropped.ravel()
    lon = lon_grid_cropped.ravel()
    landcover = landcover_cropped.ravel()
    npp = npp_cropped.ravel()
    lai = lai_cropped.ravel()
    fpar = fpar_cropped.ravel()

    valid = (
        ~np.isnan(lat)
        & ~np.isnan(lon)
        & ~np.isnan(landcover)
        & ~np.isnan(npp)
        & ~np.isnan(lai)
        & ~np.isnan(fpar)
    )

    df = pd.DataFrame(
        {
            "lat": lat[valid],
            "lon": lon[valid],
            "landcover": landcover[valid],
            "npp": npp[valid],
            "lai": lai[valid],
            "fpar": fpar[valid],
        }
    )
    df.to_csv(output_csv_path, index=False)

    del df, valid, lat, lon, landcover, npp, lai, fpar
    gc.collect()


def get_startdays():
    """Return first days of 8-day MODIS periods, matching the original script."""
    startdays = []
    for i in range(100):
        startday = i * 8 + 1
        if startday > 365:
            break
        startdays.append(startday)
    return startdays


# --------------------------------------------------------------------------------
# Main workflow

def main():
    _print(f"Creating output directory at {output_directory_main} if it doesn't exist.")
    outputdir = f"{output_directory_main}{country}/"
    os.makedirs(outputdir, exist_ok=True)

    _print("Calculating start days for 8-day periods.")
    startdays = get_startdays()

    _print(f"Looking up country ID for {country}.")
    country_id = get_country_id()

    _print(f"Scanning country raster to find bbox and required MODIS tiles for {country}.")
    required_tiles, bbox, profile = scan_country_extent_and_tiles(country_id)
    row_start, row_stop, col_start, col_stop = bbox
    output_transform = output_transform_for_bbox(profile, bbox)

    print(f"Required tiles for {country}: {required_tiles}")
    print(
        f"Country bbox for {country}: rows {row_start}:{row_stop}, "
        f"cols {col_start}:{col_stop}, shape {(row_stop - row_start, col_stop - col_start)}"
    )

    _print("Reading cropped country mask.")
    country_mask_crop = read_country_mask_crop(country_id, bbox)

    _print("Reading cropped latitude and longitude grids.")
    lat_grid_cropped, lon_grid_cropped = read_lat_lon_crops(bbox, country_mask_crop)

    for year in range(yearstart, yearend + 1):
        _print(f"Processing year {year}.")

        _print(f"Constructing cropped land cover map for {country}, year {year}.")
        landcover_cropped = construct_country_crop_map(
            year=year,
            startday=1,
            required_tiles=required_tiles,
            bbox=bbox,
            country_mask_crop=country_mask_crop,
            quantity="LandCover",
        )

        _print(f"Saving country land cover map for {country}, year {year} as GeoTIFF.")
        output_path = os.path.join(outputdir, f"{country}_LandCover_{year}.tif")
        write_geotiff(output_path, landcover_cropped, profile, output_transform)

        for startday in startdays:
            _print(f"Processing 8-day period starting on day {startday}.")

            _print("Constructing cropped NPP map.")
            npp_cropped = construct_country_crop_map(
                year=year,
                startday=startday,
                required_tiles=required_tiles,
                bbox=bbox,
                country_mask_crop=country_mask_crop,
                quantity="NPP",
            )

            _print("Constructing cropped LAI map.")
            lai_cropped = construct_country_crop_map(
                year=year,
                startday=startday,
                required_tiles=required_tiles,
                bbox=bbox,
                country_mask_crop=country_mask_crop,
                quantity="LAI",
            )

            _print("Constructing cropped FPAR map.")
            fpar_cropped = construct_country_crop_map(
                year=year,
                startday=startday,
                required_tiles=required_tiles,
                bbox=bbox,
                country_mask_crop=country_mask_crop,
                quantity="FPAR",
            )

            _print(f"Saving country NPP map for {country}, year {year}, start day {startday} as GeoTIFF.")
            output_path = os.path.join(outputdir, f"{country}_NPP_{year}_{startday:03d}.tif")
            write_geotiff(output_path, npp_cropped, profile, output_transform)

            _print(f"Saving country LAI map for {country}, year {year}, start day {startday} as GeoTIFF.")
            output_path = os.path.join(outputdir, f"{country}_LAI_{year}_{startday:03d}.tif")
            write_geotiff(output_path, lai_cropped, profile, output_transform)

            _print(f"Saving country FPAR map for {country}, year {year}, start day {startday} as GeoTIFF.")
            output_path = os.path.join(outputdir, f"{country}_FPAR_{year}_{startday:03d}.tif")
            write_geotiff(output_path, fpar_cropped, profile, output_transform)

            _print(f"Saving country data for {country}, year {year}, start day {startday} as CSV.")
            output_csv_path = os.path.join(outputdir, f"{country}_data_{year}_{startday:03d}.csv")
            write_country_csv(
                output_csv_path,
                lat_grid_cropped,
                lon_grid_cropped,
                landcover_cropped,
                npp_cropped,
                lai_cropped,
                fpar_cropped,
            )

            del npp_cropped, lai_cropped, fpar_cropped
            gc.collect()

        del landcover_cropped
        gc.collect()

    del country_mask_crop, lat_grid_cropped, lon_grid_cropped
    gc.collect()

    _print("Finished.")


if __name__ == "__main__":
    main()
