
from pyhdf.SD import SD, SDC
import numpy as np
import os
from rasterio.transform import from_origin
import rasterio

# define tile size and resolution (MODIS 500m)
res = 463.3127165275005  # meters
ntiles_x = 36
ntiles_y = 18
npixels_per_tile_x = 2400
npixels_per_tile_y = 2400

# MODIS global origin in Sinusoidal projection
ORIGIN_X = -20015109.354
ORIGIN_Y = 10007554.677

# mapping of MODIS land cover classification schemes to names in the files
# from Table 1 of
#   https://lpdaac.usgs.gov/documents/101/MCD12_User_Guide_V6.pdf
landcover_schemes = {
    'IGBP': 'LC_Type1',
    'UMD': 'LC_Type2',
    'LAI': 'LC_Type3',
    'BGC': 'LC_Type4',
    'PFT': 'LC_Type5'
}

# location for the original land cover datafiles
input_directory = f'../data/LandCover/'


def read_landcover_from_hdf(hdf_path, scheme):
    '''Extract land cover data and metadata from a MODIS HDF4 file.'''
    sd = SD(hdf_path, SDC.READ)
    dataset = sd.select(landcover_schemes[scheme]) 
    return dataset[:]

def get_mosaic_files(year):
    '''Gets all files for a given year.'''
    start_filename = f'MCD12Q1.A{year}'
    return sorted([os.path.join(input_directory, f) for f in os.listdir(input_directory) if f.startswith(start_filename) and f.endswith('.hdf')])

def construct_global_map(year, scheme):
    '''Mosaic MODIS tiles and save as a compressed GeoTIFF.'''

    # get list of all the files for the desired year
    filenames = get_mosaic_files(year)

    # initialize an empty global map
    global_map = np.zeros((ntiles_y * npixels_per_tile_y, ntiles_x * npixels_per_tile_x), dtype=int)

    for filename in filenames:

        filename_temp = filename.replace(input_directory, '')

        itile_x = int(filename_temp.split('.')[2][1:3])
        itile_y = int(filename_temp.split('.')[2][4:6])

        data_tile = read_landcover_from_hdf(filename, scheme)

        ipixel_x = itile_x * npixels_per_tile_x
        ipixel_y = itile_y * npixels_per_tile_y

        global_map[ipixel_y:ipixel_y + npixels_per_tile_y, ipixel_x:ipixel_x + npixels_per_tile_x] = data_tile

    return global_map

def generate_year_map(year, scheme):
    '''Constructs and saves the global land cover map for a given year and classification scheme.'''

    # make the map
    global_map = construct_global_map(year, scheme)

    # create output directory if it doesn't exist
    outputdir = f'../output/landcover/{scheme}/'
    os.makedirs(outputdir, exist_ok=True)

    # save the monthly average map
    output_filename = os.path.join(outputdir, f'landcover_{year}.tif')
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

def extract_landcover(scheme='UMD', year=None):
    '''Main function for extracting land cover maps'''

    if year is not None:
        generate_year_map(year, scheme)
    else:
        for year in range(2001, 2025):
            generate_year_map(year, scheme)

extract_landcover(scheme='IGBP')
