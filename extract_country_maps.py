
'''This code extracts maps of land use, npp, lai, and fpar for a specified country and outputs them as GeoTIFFs and excel files.'''

import os
import numpy as np
from pyhdf.SD import SD, SDC
import rasterio
from rasterio.transform import from_origin
from rasterio.merge import merge
import gc
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import pandas as pd

# Country to extract
country = 'Sweden'

# Define tile size and resolution (MODIS 500m)
res = 463.3127165275005  # meters
ntiles_x = 36
ntiles_y = 18
npixels_per_tile_x = 2400
npixels_per_tile_y = 2400

# MODIS global origin in Sinusoidal projection
ORIGIN_X = -20015109.354
ORIGIN_Y = 10007554.677

# files holding country raster and legend
COUNTRY_RASTER = "../output/country_map/modis_country_map.tif"
COUNTRY_LEGEND = "../output/country_map/modis_country_legend.csv"

# location for the original land cover datafiles
input_directory_landcover = f'../data/LandCover/'
input_directory_npp = f'../data/NPP/'
input_directory_lai = f'../data/LAI_fpar/'
input_directory_fpar = f'../data/LAI_fpar/'
output_directory_main = f'../output/country_maps/'

def _print(msg):
    '''Print a message with a timestamp.'''
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

def get_mosaic_files(year, startday, quantity):
    '''Gets all files for a given year and 8-day period
    
    Parameters
    ----------
    year : int
        The year of the MODIS data.
    startday : int
        The starting day of the 8-day period.
    quantity : str
        The quantity to extract (e.g. 'LandCover', 'NPP', 'LAI', 'FPAR').'''
    
    # directory is just the year
    if quantity == 'LandCover':
        directory = input_directory_landcover
    elif quantity == 'NPP':
        directory = os.path.join(input_directory_npp, f"{year}/")
    elif quantity == 'LAI':
        directory = os.path.join(input_directory_lai, f"{year}/")
    elif quantity == 'FPAR':
        directory = os.path.join(input_directory_fpar, f"{year}/")

    # convert integer for starting day to a three character string
    # e.g. 1 -> '001'
    startday_str = f"{startday:03d}"

    # get start of filename
    if quantity == 'LandCover':
        start_filename = f"MCD12Q1.A{year}{startday_str}"
    elif quantity == 'NPP':
        start_filename = f"MOD17A2HGF.A{year}{startday_str}"
    elif quantity == 'LAI':
        start_filename = f"MCD15A2H.A{year}{startday_str}"
    elif quantity == 'FPAR':
        start_filename = f"MCD15A2H.A{year}{startday_str}"
    
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.startswith(start_filename) and f.endswith('.hdf')])

def read_data_from_hdf(hdf_path, quantity):
    '''Extract data and metadata from a MODIS HDF4 file.'''

    # quantity to extract
    if quantity == 'LandCover':
        dataset_name = 'LC_Type1'
    elif quantity == 'NPP':
        dataset_name = 'Gpp_500m'
    elif quantity == 'LAI':
        dataset_name = 'Lai_500m'
    elif quantity == 'FPAR':
        dataset_name = 'Fpar_500m'
    else:
        raise ValueError(f"Unknown quantity '{quantity}'")

    # read the file and extract some data
    sd = SD(hdf_path, SDC.READ)
    dataset = sd.select(dataset_name)
    attrs = dataset.attributes()
    scale = attrs.get("scale_factor", 0.1)
    fill = attrs.get("_FillValue", -3000)
    data = dataset[:].astype(np.float32)

    # some processing based on the quantity
    if quantity == 'LandCover':
        pass

    elif quantity == 'NPP':
        
        # do some processing to get to tC ha^-1 yr^-1
        data[data == fill] = np.nan
        data *= scale
        data = np.where(data > 3.0, np.nan, data)
        data *= 456.6

        # multiply by conversion factor of 0.47 from 3PG to get NPP from GPP
        # see just after Eqn. A4 in Forrester & Tang (2016)
        data *= 0.47

    elif quantity == 'LAI':

        # set to nan all values above 100 since that is the valid range
        # https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd15a2h-061#variables
        data = np.where(data > 100, np.nan, data)

        # apply scale factor 
        data *= 0.1
        
    elif quantity == 'FPAR':

        # set to nan all values above 100 since that is the valid range
        # https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd15a2h-061#variables
        data = np.where(data > 100, np.nan, data)

        # apply scale factor 
        data *= 0.01

    return data

def construct_global_map(year, startday, required_tiles, quantity):
    '''Mosaic MODIS tiles and save as a compressed GeoTIFF.'''

    # get the input directory based on the quantity
    if quantity == 'LandCover':
        input_directory = input_directory_landcover
    elif quantity == 'NPP':
        input_directory = input_directory_npp
    elif quantity == 'LAI':
        input_directory = input_directory_lai
    elif quantity == 'FPAR':
        input_directory = input_directory_fpar

    filenames = get_mosaic_files(year, startday, quantity)
    global_map = np.zeros((ntiles_y * npixels_per_tile_y, ntiles_x * npixels_per_tile_x), dtype=np.float32)
    for filename in filenames:
        filename_temp = filename.replace(input_directory, '')
        itile_x = int(filename_temp.split('.')[2][1:3])
        itile_y = int(filename_temp.split('.')[2][4:6])
        if (itile_x, itile_y) not in required_tiles:
            continue
        data_tile = read_data_from_hdf(filename, quantity)
        ipixel_x = itile_x * npixels_per_tile_x
        ipixel_y = itile_y * npixels_per_tile_y
        global_map[ipixel_y:ipixel_y + npixels_per_tile_y, ipixel_x:ipixel_x + npixels_per_tile_x] = data_tile
    
    return global_map


def get_required_tiles(country_mask):
    '''Get the list of tiles required for a given country based on the country mask.
    
    Returns a list of (itile_x, itile_y) pairs for tiles that contain at least some unmasked data.'''

    required_tiles = []
    
    for itile_y in range(ntiles_y):
        for itile_x in range(ntiles_x):
            ipixel_x = itile_x * npixels_per_tile_x
            ipixel_y = itile_y * npixels_per_tile_y
            
            # Extract the tile region from the country mask
            tile_mask = country_mask[ipixel_y:ipixel_y + npixels_per_tile_y, 
                                     ipixel_x:ipixel_x + npixels_per_tile_x]
            
            # Check if any part of the tile is not masked
            if np.any(tile_mask):
                required_tiles.append((itile_x, itile_y))
    
    return required_tiles

def make_country_mask():
    '''Makes a mask on the MODIS global grid for the specified country.'''
    
    # get the integer ID for the country from the legend file
    country_id = None
    with open(COUNTRY_LEGEND, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2 and row[1] == country:
                country_id = int(row[0])
                break
    if country_id is None:
        raise ValueError(f"Country '{country}' not found in {COUNTRY_LEGEND}")

    # read the raster file
    with rasterio.open(COUNTRY_RASTER) as src:
        country_data = src.read(1)
        profile = src.profile.copy()
    
    # make the mask
    country_mask = country_data == country_id

    return country_mask, profile

def crop_map_to_country(map_to_crop, country_mask, profile):
    '''Crops a global map to the specified country using the country mask.'''
    rows = np.where(np.any(country_mask, axis=1))[0]
    cols = np.where(np.any(country_mask, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return np.zeros((0, 0), dtype=map_to_crop.dtype), profile['transform']
    row_start, row_stop = rows[0], rows[-1] + 1
    col_start, col_stop = cols[0], cols[-1] + 1
    country_map = map_to_crop[row_start:row_stop, col_start:col_stop]
    output_transform = profile['transform'] * rasterio.Affine.translation(col_start, row_start)
    return country_map, output_transform

def read_lat_lon_grids():
    '''Read the lat and lon grids for the global MODIS grid.'''
    lat_filename = '../data_supporting/modis500_latlon/lat_modis500.tif'
    lon_filename = '../data_supporting/modis500_latlon/lon_modis500.tif'
    with rasterio.open(lat_filename) as src:
        lat_grid = src.read(1)
    with rasterio.open(lon_filename) as src:
        lon_grid = src.read(1)
    return lat_grid, lon_grid

def main():
    '''Main function for extracting the country data.'''

    # make the output directory if it doesn't exist
    _print(f"Creating output directory at {output_directory_main} if it doesn't exist.")
    outputdir = f'{output_directory_main}{country}/'
    os.makedirs(outputdir, exist_ok=True)

    # get list of the first days in all 8-day periods in the year
    _print("Calculating start days for 8-day periods.")
    startdays = []
    for i in range(100):
        if i * 8 + 1 > 365:
            break
        startdays.append(i * 8 + 1)

    # construct the country mask
    _print(f"Constructing country mask for {country}.")
    country_mask, profile = make_country_mask()

    # work out which tiles need to be read based on the country mask
    # so not all tiles need to be read
    _print(f"Determining required tiles for {country}.")
    required_tiles = get_required_tiles(country_mask)
    print(f"Required tiles for {country}: {required_tiles}")

    # read the lat and lon grid
    _print("Reading latitude and longitude grids for the global grid and making it just for country")
    lat_grid, lon_grid = read_lat_lon_grids()
    lat_grid_masked = np.where(country_mask, lat_grid, np.nan)
    lon_grid_masked = np.where(country_mask, lon_grid, np.nan)
    lat_grid_cropped, _ = crop_map_to_country(lat_grid_masked, country_mask, profile)
    lon_grid_cropped, _ = crop_map_to_country(lon_grid_masked, country_mask, profile)
    
    # loop over years to consider (start at 2003 since that's when all data is available)
    for year in range(2003, 2024):
        _print(f"Processing year {year}.")

        # get the land cover for this year
        _print(f"Reading land cover data for year {year}.")
        global_map_landcover = construct_global_map(year, 1, required_tiles, 'LandCover')
        country_mask_landcover = np.where(country_mask, global_map_landcover, np.nan)
        country_mask_landcover_cropped, output_transform = crop_map_to_country(country_mask_landcover, country_mask, profile)

        # save the cropped country mask as a GeoTIFF using the source country profile
        _print(f"Saving country land cover map for {country}, year {year} as GeoTIFF.")
        output_path = os.path.join(outputdir, f"{country}_LandCover_{year}.tif")
        output_profile = profile.copy()
        output_profile.update(
            count=1,
            height=country_mask_landcover_cropped.shape[0],
            width=country_mask_landcover_cropped.shape[1],
            transform=output_transform,
            dtype=country_mask_landcover_cropped.dtype,
            nodata=np.nan
        )
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(country_mask_landcover_cropped, 1)

        # loop over the 8-day periods
        for startday in startdays:
            _print(f"Processing 8-day period starting on day {startday}.")

            # construct the global map for this period
            _print(f"Constructing maps.")
            global_map_npp = construct_global_map(year, startday, required_tiles, 'NPP')
            global_map_lai = construct_global_map(year, startday, required_tiles, 'LAI')
            global_map_fpar = construct_global_map(year, startday, required_tiles, 'FPAR')

            # mask the global map to get the country map
            _print(f"Masking maps to get country maps.")
            country_map_npp = np.where(country_mask, global_map_npp, np.nan)
            country_map_lai = np.where(country_mask, global_map_lai, np.nan)
            country_map_fpar = np.where(country_mask, global_map_fpar, np.nan)

            # crop the country map to the bounding box of the country
            _print(f"Cropping country maps.")
            country_map_npp_cropped, output_transform = crop_map_to_country(country_map_npp, country_mask, profile)
            country_map_lai_cropped, _ = crop_map_to_country(country_map_lai, country_mask, profile)
            country_map_fpar_cropped, _ = crop_map_to_country(country_map_fpar, country_mask, profile)

            # save the cropped country maps as GeoTIFFs using the source country profile
            output_profile = profile.copy()
            output_profile.update(
                count=1,
                transform=output_transform,
                dtype=country_map_npp_cropped.dtype,
                nodata=np.nan
            )

            _print(f"Saving country NPP map for {country}, year {year}, start day {startday} as GeoTIFF.")
            output_path = os.path.join(outputdir, f"{country}_NPP_{year}_{startday:03d}.tif")
            output_profile.update(
                height=country_map_npp_cropped.shape[0],
                width=country_map_npp_cropped.shape[1],
            )
            with rasterio.open(output_path, 'w', **output_profile) as dst:
                dst.write(country_map_npp_cropped, 1)

            _print(f"Saving country LAI map for {country}, year {year}, start day {startday} as GeoTIFF.")
            output_path = os.path.join(outputdir, f"{country}_LAI_{year}_{startday:03d}.tif")
            output_profile.update(
                height=country_map_lai_cropped.shape[0],
                width=country_map_lai_cropped.shape[1],
                dtype=country_map_lai_cropped.dtype,
            )
            with rasterio.open(output_path, 'w', **output_profile) as dst:
                dst.write(country_map_lai_cropped, 1)

            _print(f"Saving country FPAR map for {country}, year {year}, start day {startday} as GeoTIFF.")
            output_path = os.path.join(outputdir, f"{country}_FPAR_{year}_{startday:03d}.tif")
            output_profile.update(
                height=country_map_fpar_cropped.shape[0],
                width=country_map_fpar_cropped.shape[1],
                dtype=country_map_fpar_cropped.dtype,
            )
            with rasterio.open(output_path, 'w', **output_profile) as dst:
                dst.write(country_map_fpar_cropped, 1)

            # construct a DataFrame with the values for each pixel and save as CSV 
            # columns: lat, lon, landcover, npp, lai, fpar
            _print(f"Saving country data for {country}, year {year}, start day {startday} as CSV.")
            output_csv_path = os.path.join(outputdir, f"{country}_data_{year}_{startday:03d}.csv")
            df = pd.DataFrame({
                'lat': lat_grid_cropped.flatten(),
                'lon': lon_grid_cropped.flatten(),
                'landcover': country_mask_landcover_cropped.flatten(),
                'npp': country_map_npp_cropped.flatten(),
                'lai': country_map_lai_cropped.flatten(),
                'fpar': country_map_fpar_cropped.flatten(),
            })
            df = df.dropna(how='any')
            df.to_csv(output_csv_path, index=False)




if __name__ == "__main__":
    main()



