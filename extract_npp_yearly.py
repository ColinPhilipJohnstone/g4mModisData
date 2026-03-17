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
output_directory = f'../output/npp_yearly/'

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
    dataset = sd.select("PsnNet_500m")
    attrs = dataset.attributes()
    scale = attrs.get("scale_factor", 0.1)
    fill = attrs.get("_FillValue", -3000)
    data = dataset[:].astype(np.float32)

    # do some processing to get to tC ha^-1 yr^-1
    data[data == fill] = np.nan
    data *= scale
    data = np.where(data > 3.0, np.nan, data)
    data *= 456.6

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

    # make the output direcotory if it doesn't exist
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

prepare_yearly_average(2000)
