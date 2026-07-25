import os
import glob
import subprocess
import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from reproject import reproject_interp
from astropy.stats import sigma_clip
from astropy.nddata import NDData
from photutils.detection import DAOStarFinder
from photutils.background import LocalBackground
from photutils.psf import EPSFBuilder, PSFPhotometry, extract_stars

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

dark_files = glob.glob(r"D:\SARA Data\112725RM\3c273\Dark30s_Empty_*.fits")
bias_files = glob.glob(r"D:\SARA Data\112725RM\3c273\Bias_Empty_*.fits")
flat_files = glob.glob(r"D:\SARA Data\112725RM\3c273\sarm20251111_flat_JohnsonV_*.fits")[:3]
raw_science_files = glob.glob(r"D:\SARA Data\112725RM\3c273\3c273_Johnson_V_*_light.fits")[:3]

print("Dark Files Found:", len(dark_files))
print("Bias Files Found:", len(bias_files))
print("Flat Files Found:", len(flat_files))
print("Raw Science Files Found:", len(raw_science_files))
print("--------------------------------")

dark_list = [fits.open(f)[0].data for f in dark_files]
bias_list = [fits.open(f)[0].data for f in bias_files]
flat_list = [fits.open(f)[0].data for f in flat_files]

master_dark = np.median(np.array(dark_list), axis=0)
master_bias = np.median(np.array(bias_list), axis=0)

master_flat = np.median(np.array(flat_list), axis=0) - master_bias
master_flat = master_flat / np.median(master_flat)
master_flat[master_flat == 0] = 1

calibrated_science_list = []
raw_science_headers = []

for file in raw_science_files:
    hdu = fits.open(file)[0]
    raw_science_headers.append(hdu.header)
    calibrated_science_list.append((hdu.data - master_dark) / master_flat)

ref_data = calibrated_science_list[0]
ref_header = raw_science_headers[0]

hdu_ref = fits.PrimaryHDU(data=ref_data, header=ref_header)
hdu_ref.writeto("temp_ref.fits", overwrite=True)

coord = SkyCoord(f"{ref_header['RA']} {ref_header['DEC']}", unit=(u.hourangle, u.deg))
ra_deg = coord.ra.deg
dec_deg = coord.dec.deg

command_ref = (
    f"wsl solve-field temp_ref.fits "
    f"--ra {ra_deg} --dec {dec_deg} --radius 2.0 "
    f"--overwrite --no-plots"
)

print("Solving reference frame via WSL...")
subprocess.run(command_ref, shell=True)

if os.path.exists("temp_ref.new"):
    ref_wcs = WCS(fits.getheader("temp_ref.new"))
else:
    raise RuntimeError("Failed to plate-solve the reference frame.")

for temp_file in glob.glob("temp_ref*"):
    if os.path.exists(temp_file):
        os.remove(temp_file)

aligned_science_list = [ref_data]

for i in range(1, len(calibrated_science_list)):
    current_data = calibrated_science_list[i]
    current_header = raw_science_headers[i]
    
    temp_current = f"temp_frame_{i}.fits"
    hdu_curr = fits.PrimaryHDU(data=current_data, header=current_header)
    hdu_curr.writeto(temp_current, overwrite=True)
    
    print(f"\nSolving frame {i+1} of {len(calibrated_science_list)}")
    
    coord_curr = SkyCoord(f"{current_header['RA']} {current_header['DEC']}", unit=(u.hourangle, u.deg))
    
    command_loop = (
        f"wsl solve-field {temp_current} "
        f"--ra {coord_curr.ra.deg} --dec {coord_curr.dec.deg} --radius 1.0 "
        f"--overwrite --no-plots"
    )
    subprocess.run(command_loop, shell=True)
    
    solved_filename = temp_current.replace(".fits", ".new")
    
    if os.path.exists(solved_filename):
        current_wcs = WCS(fits.getheader(solved_filename))
        
        aligned_array, _ = reproject_interp(
            (current_data, current_wcs),
            output_projection=ref_wcs,
            shape_out=ref_data.shape
        )
        aligned_science_list.append(aligned_array)
    else:
        print(f"Warning: Frame {i+1} failed to solve locally. Skipping.")
        
    for temp_file in glob.glob(f"temp_frame_{i}*"):
        if os.path.exists(temp_file):
            os.remove(temp_file)

aligned_cube = np.array(aligned_science_list, dtype=float)
aligned_cube[~np.isfinite(aligned_cube)] = np.nan

clipped_cube = sigma_clip(
    aligned_cube,
    sigma=3.0,
    maxiters=5,
    cenfunc='median',
    stdfunc='std',
    axis=0,
    masked=True
)

final_img_stacked = np.ma.mean(clipped_cube, axis=0).filled(np.nan)

bg_level = np.nanmedian(final_img_stacked)
noise = np.nanstd(final_img_stacked)

final_img_stacked = np.nan_to_num(final_img_stacked, nan=bg_level)

print("--------------------------------")
print("Typical Background Level:", bg_level)
print("Typical Noise Level:", noise)

psf_finder = DAOStarFinder(threshold=5.0 * noise, fwhm=5.0)
star_catalog = psf_finder(final_img_stacked)

if star_catalog is not None and len(star_catalog) > 0:
    nddata = NDData(data=final_img_stacked)
    
    edge_margin = 15
    h, w = final_img_stacked.shape
    valid_stars = star_catalog[
        (star_catalog['xcentroid'] > edge_margin) &
        (star_catalog['xcentroid'] < w - edge_margin) &
        (star_catalog['ycentroid'] > edge_margin) &
        (star_catalog['ycentroid'] < h - edge_margin)
    ]
    
    stars_tbl = valid_stars['xcentroid', 'ycentroid']
    stars_tbl.rename_column('xcentroid', 'x')
    stars_tbl.rename_column('ycentroid', 'y')

    stars = extract_stars(nddata, stars_tbl, size=15)
    epsf_builder = EPSFBuilder(shape=None, maxiters=10, progress_bar=False)
    epsf, fitted_stars = epsf_builder(stars)

    final_engine = PSFPhotometry(
        psf_model=epsf,
        fit_shape=(11, 11),
        finder=DAOStarFinder(threshold=1.5 * noise, fwhm=5.0),
        local_bkg_estimator=LocalBackground(inner_radius=15, outer_radius=25),
        aperture_radius=5.0
    )

    phot_table = final_engine(data=final_img_stacked)
else:
    phot_table = None

print("--------------------------------")
if phot_table is not None and len(phot_table) > 0:
    print("Found", len(phot_table), "total sources.")
    print("--------------------------------")
    print(phot_table['id', 'x_fit', 'y_fit', 'flux_fit', 'flux_err'])
else:
    print("Warning: Photometry returned no sources or failed to parse data array.")
print("--------------------------------")

plt.imshow(final_img_stacked, cmap='gray', vmin=bg_level + (0.5 * noise), vmax=bg_level + (10 * noise))
plt.show()
