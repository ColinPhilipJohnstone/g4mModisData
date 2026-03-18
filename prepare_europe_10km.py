import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from pyproj import CRS
import os
import numpy as np
from scipy import ndimage
from datetime import datetime
from rasterio.warp import reproject, Resampling
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# some params
outputdir = 'Europe_10km'
months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
landcover_types = {
    'water': 0,
    'evergreen_needleleaf_forest': 1,
    'evergreen_broadleaf_forest': 2,
    'deciduous_needleleaf_forest': 3,
    'deciduous_broadleaf_forest': 4,
    'mixed_forest': 5,
    'closed_shrubland': 6,
    'open_shrubland': 7,
    'woody_savannah': 8,
    'savannah': 9,
    'grassland': 10,
    'cropland': 12,
    'urban': 13,
    'barren': 16
}

# the file holding the information about the target grid
target_grid_file = '../data_supporting/europe_10km/pixel_mask.tif'

def _print(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def crop_modis_europe(input_tif: str, output_tif: str, europe_lonlat_bounds=(-12.0, 34.0, 45.0, 72.0)):
    '''Crop a MODIS sinusoidal GeoTIFF to a Europe subset and save as a new GeoTIFF.'''
    
    # Standard MODIS sinusoidal CRS, used only if the source file has no CRS metadata
    modis_sinu = CRS.from_proj4(
        "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext +units=m +no_defs"
    )

    lon_min, lat_min, lon_max, lat_max = europe_lonlat_bounds

    with rasterio.open(input_tif) as src:
        src_crs = src.crs if src.crs is not None else modis_sinu

        # Transform Europe bounding box from lon/lat (EPSG:4326) into source CRS
        # densify_pts makes the transformed bounds safer for curved projections
        left, bottom, right, top = transform_bounds(
            "EPSG:4326",
            src_crs,
            lon_min,
            lat_min,
            lon_max,
            lat_max,
            densify_pts=21
        )

        # Build crop window from projected bounds
        window = from_bounds(left, bottom, right, top, transform=src.transform)

        # Clip window to raster extent
        window = window.round_offsets().round_lengths()
        window = window.intersection(
            rasterio.windows.Window(0, 0, src.width, src.height)
        )

        # Read cropped data
        data = src.read(window=window)

        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "height": int(window.height),
            "width": int(window.width),
            "transform": rasterio.windows.transform(window, src.transform)
        })

        # Write output
        with rasterio.open(output_tif, "w", **out_meta) as dst:
            dst.write(data)

def make_masks(input_tif, output_dir, year):
    '''Creates masks for each land cover type.'''

    # read land cover map for this year
    with rasterio.open(input_tif) as src:
        landcover_data = src.read(1)
        meta = src.meta

    # loop over types and make mask for each
    for type_name, type_value in landcover_types.items():
        mask_data = (landcover_data == type_value).astype(rasterio.uint8)

        # write mask to file
        output_tif = os.path.join(output_dir, f"mask_{type_name}_{year}.tif")
        with rasterio.open(output_tif, "w", **meta) as dst:
            dst.write(mask_data, 1)

    # also do generic forest mask
    forest_mask_data = ((landcover_data >= 1) & (landcover_data <= 5)).astype(rasterio.uint8)
    output_tif = os.path.join(output_dir, f"mask_forest_{year}.tif")
    with rasterio.open(output_tif, "w", **meta) as dst:
        dst.write(forest_mask_data, 1)

def aggregate_maps(input_tif_mask, input_tif_npp, output_tif_area, output_tif_npp, factor=20):
    '''Aggregates the cropped maps to approx 10 km resolution and puts in the output directory'''

    # read mask
    with rasterio.open(input_tif_mask) as src:
        mask = src.read(1)
        meta = src.meta
    
    # read npp map
    with rasterio.open(input_tif_npp) as src:
        npp = src.read(1)
        meta = src.meta

    # do the aggregation of the mask
    # using sum to get number of pixels in aggegated pixels that are covered in this type
    # then convert to area in ha (original pixels are 463.312x463.312 m^2 = 21.466 ha)
    h, w = mask.shape
    mask_agg = mask[:h//factor*factor, :w//factor*factor].reshape(h//factor, factor, w//factor, factor).sum(axis=(1, 3))
    mask_agg *= 21.466

    # remove missing values and convert units
    # npp[npp > 3.0] = np.nan  # the highest values are missing values

    # do the aggregation of the npp map, using mean to get average npp in each aggregated pixel
    npp_masked = np.where(mask, npp, np.nan)
    npp_agg = npp_masked[:h//factor*factor, :w//factor*factor].reshape(h//factor, factor, w//factor, factor)
    npp_agg = np.nanmean(npp_agg, axis=(1, 3))

    # write the aggregated maps
    meta.update({
        "height": mask_agg.shape[0],
        "width": mask_agg.shape[1],
        "transform": rasterio.Affine(meta["transform"].a * factor, 0, meta["transform"].c, 0, meta["transform"].e * factor, meta["transform"].f)
    })

    with rasterio.open(output_tif_area, "w", **meta) as dst:
        dst.write(mask_agg.astype(rasterio.float32), 1)

    with rasterio.open(output_tif_npp, "w", **meta) as dst:
        dst.write(npp_agg.astype(rasterio.float32), 1)


def reproject_maps(input_tif, output_tif):
    '''Reprojects a map to the desired grid'''

    with rasterio.open(target_grid_file) as target_src:
        target_crs = target_src.crs
        target_transform = target_src.transform
        target_width = target_src.width
        target_height = target_src.height
        target_meta = target_src.meta.copy()
    
    with rasterio.open(input_tif) as src:
        data = src.read()
        
        nodata_val = np.nan

        # Update metadata for reprojection
        target_meta.update({
            "crs": target_crs,
            "transform": target_transform,
            "width": target_width,
            "height": target_height,
            "dtype": "float32",
            "nodata": nodata_val,
        })
        
        # Reproject and write
        with rasterio.open(output_tif, "w", **target_meta) as dst:
            for band_idx in range(src.count):
                reproject(
                    source=rasterio.band(src, band_idx + 1),
                    destination=rasterio.band(dst, band_idx + 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=nodata_val,      # critical
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    dst_nodata=nodata_val,      # critical
                    init_dest_nodata=True,
                    resampling=Resampling.bilinear
                )

        # Apply pixel mask to the reprojected output
        with rasterio.open(target_grid_file) as target_src:
            pixel_mask = target_src.read(1)

        with rasterio.open(output_tif, "r+") as dst:
            for band_idx in range(1, dst.count + 1):
                data = dst.read(band_idx)
                data = np.where(pixel_mask == 1, data, nodata_val)
                dst.write(data.astype(np.float32), band_idx)


def compare_neumann():
    '''Compares the prepared map to the one from Neumann et al. 1996'''

    for year in range(2001, 2002):
            
        with rasterio.open(f'../data_supporting/MODIS_EURO/Europe_10km/EU_NPP_{year}_reprojected.tif') as src:
            data_neumann = src.read(1)
        
        with rasterio.open(f'../output/Europe_10km_2/reprojected/forest_npp_{year}.tif') as src:
            data = src.read(1)

        # data *= 0.47


        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        # Map 1: Neumann data
        im1 = axes[0].imshow(data_neumann, cmap='viridis')
        axes[0].set_title('Neumann et al. 1996')
        plt.colorbar(im1, ax=axes[0])

        # Map 2: Our data
        im2 = axes[1].imshow(data, cmap='viridis')
        axes[1].set_title(f'Prepared Data ({year})')
        plt.colorbar(im2, ax=axes[1])

        # Scatter plot with density coloring
        valid = ~(np.isnan(data_neumann) | np.isnan(data))
        x = data_neumann[valid]
        y = data[valid]
        z = gaussian_kde(np.vstack([x, y]))(np.vstack([x, y]))

        scatter = axes[2].scatter(x, y, c=z, s=1, cmap='viridis')
        axes[2].plot([x.min(), x.max()], [x.min(), x.max()], 'r--', lw=2, label='y=x')
        axes[2].plot([x.min(), x.max()], [2*x.min(), 2*x.max()], 'g--', lw=2, label='y=2x')
        axes[2].legend()
        axes[2].set_xlabel('Neumann et al. 1996')
        axes[2].set_ylabel(f'Prepared Data ({year})')
        axes[2].set_title('Comparison')
        plt.colorbar(scatter, ax=axes[2], label='Density')

        # Add fourth panel with histograms
        axes[3].hist(x, bins=30, alpha=0.6, label='Neumann et al. 1996', color='blue', density=True)
        axes[3].hist(y, bins=30, alpha=0.6, label=f'Prepared Data ({year})', color='orange', density=True)
        axes[3].set_xlabel('NPP Value')
        axes[3].set_ylabel('Frequency')
        axes[3].set_title('Histograms')
        axes[3].legend()

        plt.tight_layout()
        plt.savefig(f'../output/Europe_10km_2/comparison_{year}.png', dpi=150)
        plt.close()



def setup_europe_10km():
    '''Sets up the data for the 10km Europe map'''

    # create the output directories if they don't exist
    outputdir = f'../output/Europe_10km/'
    os.makedirs(outputdir, exist_ok=True)

    # the years to prepare
    years = range(2001, 2002)

    #-----------------------------------------
    # CROPPING

    # crop the landcover maps
    outputdir2 = f'{outputdir}landcover_cropped/'
    os.makedirs(outputdir2, exist_ok=True)
    for year in years:
        _print(f"Cropping land cover map for {year}...")
        input_tif = f'../output/landcover/UMD/landcover_{year}.tif'
        output_tif = f'{outputdir2}landcover_{year}.tif'
        crop_modis_europe(input_tif, output_tif)

    # crop the yearly NPP maps
    outputdir2 = f'{outputdir}npp_yearly_cropped/'
    os.makedirs(outputdir2, exist_ok=True)
    for year in years:
        _print(f"Cropping NPP map for {year}...")
        input_tif = f'../output/npp_yearly/npp_{year}.tif'
        output_tif = f'{outputdir2}npp_{year}.tif'
        crop_modis_europe(input_tif, output_tif)

    # crops the monthly maps
    pass

    #-----------------------------------------
    # MAKE MASKS

    outputdir2 = f'{outputdir}land_cover_masks/'
    os.makedirs(outputdir2, exist_ok=True)
    for year in years:
        input_tif = f'{outputdir}landcover_cropped/landcover_{year}.tif'
        make_masks(input_tif, outputdir2, year)

    #-----------------------------------------
    # AGGREGATE TO 10KM

    outputdir2 = f'{outputdir}aggregated/'
    os.makedirs(outputdir2, exist_ok=True)
    for year in years:
        input_tif_mask = f'{outputdir}land_cover_masks/mask_forest_{year}.tif'
        input_tif_npp = f'{outputdir}npp_yearly_cropped/npp_{year}.tif'
        output_tif_area = f'{outputdir2}forest_area_{year}.tif'
        output_tif_npp = f'{outputdir2}forest_npp_{year}.tif'
        aggregate_maps(input_tif_mask, input_tif_npp, output_tif_area, output_tif_npp)

    #-----------------------------------------
    # REPROJECT

    outputdir2 = f'{outputdir}reprojected/'
    os.makedirs(outputdir2, exist_ok=True)
    for year in years:
        input_tif = f'{outputdir}aggregated/forest_npp_{year}.tif'
        output_tif = f'{outputdir2}forest_npp_{year}.tif'
        reproject_maps(input_tif, output_tif)


# setup_europe_10km()
compare_neumann()

