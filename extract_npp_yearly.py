import os
import numpy as np
from pyhdf.SD import SD, SDC
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
input_directory = f'../data/NPP/'
output_directory_main = f'../output/'

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
    start_filename = f"MOD17A2HGF.A{year}{startday_str}"

    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(start_filename) and f.endswith('.hdf')])

def read_npp_from_hdf(hdf_path):
    '''Extract NPP data and metadata from a MODIS HDF4 file.'''

    # read the file and extract some data
    sd = SD(hdf_path, SDC.READ)
    dataset = sd.select('Gpp_500m')
    attrs = dataset.attributes()
    scale = attrs.get("scale_factor", 0.1)
    fill = attrs.get("_FillValue", -3000)
    data = dataset[:].astype(np.float32)

    # do some processing to get to tC ha^-1 yr^-1
    data[data == fill] = np.nan
    data *= scale
    data = np.where(data > 3.0, np.nan, data)
    data *= 456.6

    # multiply by conversion factor of 0.47 from 3PG to get NPP from GPP
    # see just after Eqn. A4 in Forrester & Tang (2016)
    data *= 0.47

    return data

def construct_global_map(year, startday):
    '''Mosaic MODIS tiles and save as a compressed GeoTIFF.'''

    filenames = get_mosaic_files(year, startday)
    global_map = np.zeros((ntiles_y * npixels_per_tile_y, ntiles_x * npixels_per_tile_x), dtype=np.float32)
    for filename in filenames:
        filename_temp = filename.replace(input_directory, '')
        itile_x = int(filename_temp.split('.')[2][1:3])
        itile_y = int(filename_temp.split('.')[2][4:6])
        data_tile = read_npp_from_hdf(filename)
        ipixel_x = itile_x * npixels_per_tile_x
        ipixel_y = itile_y * npixels_per_tile_y
        global_map[ipixel_y:ipixel_y + npixels_per_tile_y, ipixel_x:ipixel_x + npixels_per_tile_x] = data_tile
    
    return global_map

def prepare_yearly_average(year):
    '''Prepares global maps of yearly average NPP data for a given year.'''
    
    # check year is provided 
    if year is None:
        raise ValueError("Year must be specified for yearly average preparation.")
    
    # get the output directory for the quantity
    output_directory = os.path.join(output_directory_main, 'npp_yearly/')

    # make the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # get list of the first days in all 8-day periods in the year
    startdays = []
    for i in range(100):
        if i * 8 + 1 > 365:
            break
        startdays.append(i * 8 + 1)

    # calculate and output yearly average maps
    global_map = None
    for startday in startdays:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing year {year}, start day {startday}...")

        global_maps_8days = construct_global_map(year, startday)
        if global_map is None:
            global_map = global_maps_8days
        else:
            global_map += global_maps_8days

    # divide by number of 8-day periods to get yearly average
    global_map /= len(startdays)

    # save the yearly average map
    output_filename = os.path.join(output_directory, f"npp_{year}.tif")
    out_meta = {
        "driver": "GTiff",
        "height": global_map.shape[0],
        "width": global_map.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs",  # MODIS Sinusoidal
        "transform": from_origin(ORIGIN_X, ORIGIN_Y, res, res),
        "compress": "deflate",
        "tiled": True
    }
    with rasterio.open(output_filename, "w", **out_meta) as dest:
        dest.write(global_map, 1)

def prepare_yearly_average_all():
    '''Prepares global maps of yearly average NPP data for all years.'''
    for year in range(2001, 2025):
        prepare_yearly_average(year)

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
    '''Prepares global maps of monthly average NPP data for a given year.'''
    
    # check year is provided
    if year is None:
        raise ValueError("Year must be specified for monthly average preparation.")

    # create directory for this year if it does not exist
    output_directory = os.path.join(output_directory_main, 'npp_monthly/')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # now do the same for the year's subdirectory
    output_directory_year = os.path.join(output_directory, f"{year}/")
    if not os.path.exists(output_directory_year):
        os.makedirs(output_directory_year)

    # get list of the first days in all 8-day periods in the year
    startdays = []
    for i in range(100):
        if i * 8 + 1 > 365:
            break
        startdays.append(i * 8 + 1)
    
    # make dictionary holding global maps for each 8-day period
    # this will for now just be None instead of maps
    # the actual maps will be loaded and unloaded as needed
    global_maps_8day = {}
    for startday in startdays:
        global_maps_8day[startday] = None
    
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
                if global_maps_8day[startday_8day] is not None:
                    print(f"Unloading 8-day map for {startday_8day} as it does not contribute to month {month}.")
                    del global_maps_8day[startday_8day]
                    global_maps_8day[startday_8day] = None
                    gc.collect()
            else:
                # if it is not loaded, load it
                if global_maps_8day[startday_8day] is None:
                    print(f"Loading 8-day map for {startday_8day} for month {month}.")
                    global_maps_8day[startday_8day] = construct_global_map(year, startday_8day)

        # start array to hold the map for this month
        month_map = np.zeros((ntiles_y * npixels_per_tile_y, ntiles_x * npixels_per_tile_x), dtype=np.float32)

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
                month_map += global_maps_8day[startday_8day] * ndays

        # check total number of days is correct
        if ndaysmonth != month_lengths[month]:
            raise ValueError(f"Number of days in month {month} ({ndaysmonth}) does not match expected length ({month_lengths[month]}).")

        # divide by number of days in the month to get average
        if ndaysmonth > 0:
            month_map /= ndaysmonth

        # save the monthly average map
        output_filename = f"{output_directory_year}/npp_{year}_{month.lower()}.tif"
        out_meta = {
            "driver": "GTiff",
            "height": month_map.shape[0],
            "width": month_map.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs",  # MODIS Sinusoidal
            "transform": from_origin(ORIGIN_X, ORIGIN_Y, res, res),
            "compress": "deflate",
            "tiled": True
        }
        with rasterio.open(output_filename, "w", **out_meta) as dest:
            dest.write(month_map, 1)

    # loop over months again and add them up to get the yearly average
    ndays_total = 0
    yearly_map = np.zeros((ntiles_y * npixels_per_tile_y, ntiles_x * npixels_per_tile_x), dtype=np.float32)
    for month in range(12):

        # read month map
        month_name = list(month_ranges.keys())[month]
        month_filename = f"{output_directory_year}/npp_{year}_{month_name.lower()}.tif"
        with rasterio.open(month_filename) as src:
            month_map = src.read(1)
        
        # add to yearly map weighted by number of days in the month
        ndays_total += month_lengths[month_name]
        yearly_map += month_map * month_lengths[month_name]

    # divide by number of days in the year to get average
    yearly_map /= ndays_total

    # save the yearly average map
    output_filename = os.path.join(output_directory_year, f"npp_{year}.tif")
    out_meta = {
        "driver": "GTiff",
        "height": yearly_map.shape[0],
        "width": yearly_map.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs",  # MODIS Sinusoidal
        "transform": from_origin(ORIGIN_X, ORIGIN_Y, res, res),
        "compress": "deflate",
        "tiled": True
    }
    with rasterio.open(output_filename, "w", **out_meta) as dest:
        dest.write(yearly_map, 1)

# prepare_yearly_average_all()
for year in range(2002, 2025):
    prepare_monthly_averages(year)
