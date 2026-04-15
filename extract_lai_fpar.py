import os
import numpy as np
from pyhdf.SD import SD, SDC, HDF4Error
import rasterio
from rasterio.transform import from_origin
from rasterio.merge import merge
import gc
from datetime import datetime
import matplotlib.pyplot as plt

# Define tile size and resolution (MODIS 500m)
res = 463.3127165275005  # meters
ntiles_x = 36
ntiles_y = 18
npixels_per_tile_x = 2400
npixels_per_tile_y = 2400

# MODIS global origin in Sinusoidal projection
ORIGIN_X = -20015109.354
ORIGIN_Y = 10007554.677

# location for the original land cover datafiles
# input_directory = '../data/LAI/'
input_directory = '../data/LAI_fpar/'
output_directory_main = '../output/'

def get_mosaic_files(year, startday):
    '''Gets all files for a given year and 8-day period
    
    Parameters
    ----------
    year : int
        The year of the MODIS data.
    startday : int
        The starting day of the 8-day period.'''
    
    # directory is just the year
    directory = os.path.join(input_directory, f"{year}/")
    
    # convert integer for starting day to a three character string
    # e.g. 1 -> '001'
    startday_str = f"{startday:03d}"

    # get start of filename
    start_filename = f"MCD15A2H.A{year}{startday_str}"
    print(f"Looking for files in {directory} starting with {start_filename}..." )

    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(start_filename) and f.endswith('.hdf')])
 
def read_lai_fpar_from_hdf(hdf_path):
    '''Extract LAI and FPAR data from a MODIS HDF4 file.'''

    hdf_path = os.path.abspath(hdf_path)

    if not os.path.isfile(hdf_path):
        raise FileNotFoundError(f"Missing HDF file: {hdf_path}")
    if hdf_path.lower().endswith(".xml"):
        raise ValueError(f"Not an HDF4 data file (XML sidecar): {hdf_path}")

    size = os.path.getsize(hdf_path)
    if size == 0:
        raise OSError(f"Empty file: {hdf_path}")

    if not os.access(hdf_path, os.R_OK):
        raise PermissionError(f"File is not readable: {hdf_path}")

    try:
        sd = SD(hdf_path, SDC.READ)
    except HDF4Error as e:
        raise RuntimeError(f"HDF4 open failed for: {hdf_path}") from e

    try:
        data_lai = sd.select("Lai_500m").get().astype(np.float32)
        data_fpar = sd.select("Fpar_500m").get().astype(np.float32)
    finally:
        sd.end()

    # set to nan all values above 100 since that is the valid range
    # https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd15a2h-061#variables
    data_lai = np.where(data_lai > 100, np.nan, data_lai)
    data_fpar = np.where(data_fpar > 100, np.nan, data_fpar)

    # apply scale factor 
    data_lai *= 0.1
    data_fpar *= 0.01

    return data_lai, data_fpar

def construct_global_map(year, startday):
    '''Mosaic MODIS tiles and save as a compressed GeoTIFF.'''

    filenames = get_mosaic_files(year, startday)
    print(f'Reading {len(filenames)} files for year {year} and start day {startday}...')

    global_map_lai = np.zeros((ntiles_y * npixels_per_tile_y, ntiles_x * npixels_per_tile_x), dtype=np.float32)
    global_map_fpar = np.zeros((ntiles_y * npixels_per_tile_y, ntiles_x * npixels_per_tile_x), dtype=np.float32)
    skipped_files = []
    for filename in filenames:
        filename_temp = filename.replace(input_directory, '')
        itile_x = int(filename_temp.split('.')[2][1:3])
        itile_y = int(filename_temp.split('.')[2][4:6])
        try:
            data_lai, data_fpar = read_lai_fpar_from_hdf(filename)
        except Exception as e:
            skipped_files.append((filename, str(e)))
            continue
        ipixel_x = itile_x * npixels_per_tile_x
        ipixel_y = itile_y * npixels_per_tile_y
        global_map_lai[ipixel_y:ipixel_y + npixels_per_tile_y, ipixel_x:ipixel_x + npixels_per_tile_x] = data_lai
        global_map_fpar[ipixel_y:ipixel_y + npixels_per_tile_y, ipixel_x:ipixel_x + npixels_per_tile_x] = data_fpar

    if skipped_files:
        print(f"Skipped {len(skipped_files)} unreadable/corrupt HDF files for year {year}, start day {startday}:")
        for bad_file, err in skipped_files:
            print(f"  - {bad_file}: {err}")
    
    return global_map_lai, global_map_fpar

def get_month_day_ranges(year):
    '''Returns a dictionary mapping month names to (first_day, last_day) integers.'''

    leap_year = year in [2000, 2004, 2008, 2012, 2016, 2020, 2024]

    month_lengths = {
        "January": 31,
        "February": 29 if leap_year else 28,
        "March": 31,
        "April": 30,
        "May": 31,
        "June": 30,
        "July": 31,
        "August": 31,
        "September": 30,
        "October": 31,
        "November": 30,
        "December": 31
    }

    day_ranges = {}
    day_counter = 1
    for month, length in month_lengths.items():
        start_day = day_counter
        end_day = start_day + length - 1
        day_ranges[month] = (start_day, end_day)
        day_counter = end_day + 1

    return month_lengths, day_ranges

def prepare_monthly_averages(year):
    '''Prepares global maps of monthly average LAI for a given year.'''
    
    # check year is provided
    if year is None:
        raise ValueError("Year must be specified for monthly average preparation.")

    # create directory for this year if it does not exist
    output_directory_lai = os.path.join(output_directory_main, 'lai_monthly/')
    if not os.path.exists(output_directory_lai):
        os.makedirs(output_directory_lai)
    output_directory_fpar = os.path.join(output_directory_main, 'fpar_monthly/')
    if not os.path.exists(output_directory_fpar):
        os.makedirs(output_directory_fpar)

    # now do the same for the year's subdirectory
    output_directory_year_lai = os.path.join(output_directory_lai, f"{year}/")
    if not os.path.exists(output_directory_year_lai):
        os.makedirs(output_directory_year_lai)
    output_directory_year_fpar = os.path.join(output_directory_fpar, f"{year}/")
    if not os.path.exists(output_directory_year_fpar):
        os.makedirs(output_directory_year_fpar)

    # get list of the first days in all 8-day periods in the year
    startdays = []
    for i in range(100):
        if i * 8 + 1 > 365:
            break
        startdays.append(i * 8 + 1)
    
    # make dictionary holding global maps for each 8-day period
    # this will for now just be None instead of maps
    # the actual maps will be loaded and unloaded as needed
    global_maps_8day_lai = {}
    global_maps_8day_fpar = {}
    for startday in startdays:
        global_maps_8day_lai[startday] = None
        global_maps_8day_fpar[startday] = None
    
    # get start and end days for each month
    month_lengths, month_ranges = get_month_day_ranges(year)

    # calculate and output monthly average maps
    for month, (startday_month, endday_month) in month_ranges.items():

        # loop over start days and load the ones that are needed for this month if not loaded
        # and unload the ones that are loaded and not needed
        for startday_8day in startdays:
            endday_8day = startday_8day + 7
            
            # get number of days in this 8-day period that fall within the month
            ndays = max(0, min(endday_8day, endday_month) - max(startday_8day, startday_month) + 1)

            # if ndays is zero then this 8-day period does not contribute to the month
            # so unload it if it is loaded
            if ndays == 0:
                if global_maps_8day_lai[startday_8day] is not None:
                    print(f"Unloading 8-day map for {startday_8day} as it does not contribute to month {month}.")
                    del global_maps_8day_lai[startday_8day]
                    global_maps_8day_lai[startday_8day] = None
                    del global_maps_8day_fpar[startday_8day]
                    global_maps_8day_fpar[startday_8day] = None
                    gc.collect()
            else:
                # if it is not loaded, load it
                if global_maps_8day_lai[startday_8day] is None:
                    print(f"Loading 8-day map for {startday_8day} for month {month}.")
                    global_maps_8day_lai[startday_8day], global_maps_8day_fpar[startday_8day] = construct_global_map(year, startday_8day)

        # start array to hold the map for this month
        month_map_lai = np.zeros((ntiles_y * npixels_per_tile_y, ntiles_x * npixels_per_tile_x), dtype=np.float32)
        month_map_fpar = np.zeros((ntiles_y * npixels_per_tile_y, ntiles_x * npixels_per_tile_x), dtype=np.float32)

        # get the 8-day periods that fall within this month
        ndaysmonth = 0 # number of days in the month 
        for startday_8day in startdays:
            endday_8day = startday_8day + 7

            # calculate how many days in this 8-day period fall within the month
            ndays = max(0, min(endday_8day, endday_month) - max(startday_8day, startday_month) + 1)
            ndaysmonth += ndays

            # print(f"Processing month {month}: {startday_month} to {endday_month}, 8-day period {startday_8day} has {ndays} days.")

            # add this map weighted by number of days
            if ndays > 0:
                month_map_lai += global_maps_8day_lai[startday_8day] * ndays
                month_map_fpar += global_maps_8day_fpar[startday_8day] * ndays

        # check total number of days is correct
        if ndaysmonth != month_lengths[month]:
            raise ValueError(f"Number of days in month {month} ({ndaysmonth}) does not match expected length ({month_lengths[month]}).")

        # divide by number of days in the month to get average
        if ndaysmonth > 0:
            month_map_lai /= ndaysmonth
            month_map_fpar /= ndaysmonth

        # save the monthly average map of lai
        output_filename = f"{output_directory_year_lai}/lai_{year}_{month.lower()}.tif"
        out_meta = {
            "driver": "GTiff",
            "height": month_map_lai.shape[0],
            "width": month_map_lai.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs",  # MODIS Sinusoidal
            "transform": from_origin(ORIGIN_X, ORIGIN_Y, res, res),
            "compress": "deflate",
            "tiled": True
        }
        with rasterio.open(output_filename, "w", **out_meta) as dest:
            dest.write(month_map_lai, 1)

        # save the monthly average map of fpar
        output_filename = f"{output_directory_year_fpar}/fpar_{year}_{month.lower()}.tif"
        out_meta = {
            "driver": "GTiff",
            "height": month_map_fpar.shape[0],
            "width": month_map_fpar.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs",  # MODIS Sinusoidal
            "transform": from_origin(ORIGIN_X, ORIGIN_Y, res, res),
            "compress": "deflate",
            "tiled": True
        }
        with rasterio.open(output_filename, "w", **out_meta) as dest:
            dest.write(month_map_fpar, 1)

for year in range(2004, 2025):
    prepare_monthly_averages(year)
