import os
import glob
import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from astroquery.astrometry_net import AstrometryNet
from reproject import reproject_interp
from photutils.detection import DAOStarFinder
from photutils.background import LocalBackground
from photutils.psf import CircularGaussianPSF, PSFPhotometry

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

load_dotenv()
ast = AstrometryNet()
ast.api_key = os.getenv("ASTROMETRY_API_KEY")

dark_files = glob.glob(r"D:\your\file\here_*.fits")
bias_files = glob.glob(r"D:\your\file\here_*.fits")
flat_files = glob.glob(r"D:\your\file\here_*.fits")
raw_science_files = glob.glob(r"D:\your\file\here_*.fits")

dark_list = []
bias_list = []
flat_list = []

for dark_file in dark_files:
    dark_list.append(fits.open(dark_file)[0].data)
for bias_file in bias_files:
    bias_list.append(fits.open(bias_file)[0].data)
for flat_file in flat_files:
    flat_list.append(fits.open(flat_file)[0].data)

dark_cube = np.array(dark_list)
bias_cube = np.array(bias_list)
flat_cube = np.array(flat_list)

master_dark = np.median(dark_cube, axis=0)
master_bias = np.median(bias_cube, axis=0)

master_flat = np.median(flat_cube, axis=0) - master_bias
master_flat = master_flat / np.median(master_flat)
master_flat[master_flat == 0] = 1

calibrated_science_list = []
raw_science_headers = []

for file in raw_science_files:
    hdu = fits.open(file)[0]
    raw_science_headers.append(hdu.header)
    calibrated_science_list.append((hdu.data - master_dark) / master_flat)

# Solving Reference Frame
ref_data = calibrated_science_list[0]
ref_header = raw_science_headers[0]

hdu_ref = fits.PrimaryHDU(data=ref_data, header=ref_header)
hdu_ref.writeto("temp_ref.fits", overwrite=True)

ra_str = ref_header['RA']
dec_str = ref_header['DEC']
coord_str = f"{ra_str} {dec_str}"

coord = SkyCoord(coord_str, unit=(u.hourangle, u.deg))
ra_deg = coord.ra.deg
dec_deg = coord.dec.deg

ref_header_solved = ast.solve_from_image(
    "temp_ref.fits", 
    center_ra=ra_deg, 
    center_dec=dec_deg, 
    radius=2.0, 
    solve_timeout=300)

# SAFETY 1: Stop immediately if the reference frame fails to solve
if ref_header_solved is None:
    os.remove("temp_ref.fits")
    raise RuntimeError("Reference frame failed to plate-solve")

ref_wcs = WCS(ref_header_solved)
os.remove("temp_ref.fits")

# Alignment Loop
aligned_science_list = [ref_data]
for i in range(1, len(calibrated_science_list)):
    current_data = calibrated_science_list[i]
    current_header = raw_science_headers[i]
    
    temp_current = f"temp_frame_{i}.fits"
    hdu_curr = fits.PrimaryHDU(data=current_data, header=current_header)
    hdu_curr.writeto(temp_current, overwrite=True)
    print(f"Solving frame {i+1} of {len(calibrated_science_list)}...")
    
    coord = SkyCoord(f"{current_header['RA']} {current_header['DEC']}", unit=(u.hourangle, u.deg))
    current_header_solved = ast.solve_from_image(temp_current, center_ra=coord.ra.deg, center_dec=coord.dec.deg, radius=2.0, solve_timeout=300)
    
    # SAFETY 2: Skip any frame that fails to solve instead of crashing the whole stack
    if current_header_solved is None:
        print(f"Warning: Frame {i+1} failed to solve. Skipping.")
        os.remove(temp_current)
        continue
        
    current_wcs = WCS(current_header_solved)
    aligned_array, _ = reproject_interp((current_data, current_wcs), output_projection=ref_wcs, shape_out=ref_data.shape)
    aligned_science_list.append(aligned_array)
    os.remove(temp_current)

final_img_stacked = np.median(np.array(aligned_science_list), axis=0)
bg_level = np.median(final_img_stacked)
noise = np.std(final_img_stacked)

print("--------------------------------")
print("Typical Background Level:", bg_level)
print("Typical Noise Level:", noise)

# Photometry Engine
final_engine = PSFPhotometry(
    psf_model=CircularGaussianPSF(fwhm=5.0),
    fit_shape=(11, 11),
    finder=DAOStarFinder(threshold=1.5 * noise, fwhm=5.0),
    local_bkg_estimator=LocalBackground(inner_radius=15, outer_radius=25),
    aperture_radius=5.0)

phot_table = final_engine(data=final_img_stacked)

print("--------------------------------")
print("Found", len(phot_table), "total sources.")
print("--------------------------------")

print(phot_table['id', 'x_fit', 'y_fit', 'flux_fit', 'flux_err'])

plt.imshow(final_img_stacked, cmap='gray', vmin=bg_level + (0.5 * noise), vmax=bg_level + (10 * noise))
plt.show()
