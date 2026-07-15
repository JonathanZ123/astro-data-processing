import os
import glob
import numpy as np
import astropy.io.fits as fits
import astropy.units as u
from astropy.coordinates import SkyCoord
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

dark_files = glob.glob(r"D:\SARA Data\112725RM\3c273\Dark30s_Empty_*.fits")
bias_files = glob.glob(r"D:\SARA Data\112725RM\3c273\Bias_Empty_*.fits")
flat_files = glob.glob(r"D:\SARA Data\112725RM\3c273\sarm20251111_flat_JohnsonV_*.fits")
raw_science_files = glob.glob(r"D:\SARA Data\112725RM\3c273\3c273_Johnson_V_*_light.fits")

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

calibrated_science = []
raw_headers = []
for file in raw_science_files:
    hdu = fits.open(file)[0]
    raw_headers.append(hdu.header)
    calibrated_science.append((hdu.data - master_dark) / master_flat)

ref_data = calibrated_science[0]
ref_header = raw_headers[0]
fits.PrimaryHDU(data=ref_data, header=ref_header).writeto("temp_ref.fits", overwrite=True)

coord = SkyCoord(f"{ref_header['RA']} {ref_header['DEC']}", unit=(u.hourangle, u.deg))
ref_wcs = ast.solve_from_image("temp_ref.fits", center_ra=coord.ra.deg, center_dec=coord.dec.deg, radius=2.0, solve_timeout=300)
os.remove("temp_ref.fits")

aligned_science_list = [ref_data]
for i in range(1, len(calibrated_science)):
    current_data = calibrated_science[i]
    current_header = raw_headers[i]
    
    temp_file = f"temp_frame_{i}.fits"
    fits.PrimaryHDU(data=current_data, header=current_header).writeto(temp_file, overwrite=True)
    
    curr_coord = SkyCoord(f"{current_header['RA']} {current_header['DEC']}", unit=(u.hourangle, u.deg))
    current_wcs = ast.solve_from_image(temp_file, center_ra=curr_coord.ra.deg, center_dec=curr_coord.dec.deg, radius=2.0, solve_timeout=300)
    
    aligned_array, _ = reproject_interp((current_data, current_wcs), output_projection=ref_wcs, shape_out=ref_data.shape)
    aligned_science_list.append(aligned_array)
    os.remove(temp_file)

final_img_stacked = np.median(np.array(aligned_science_list), axis=0)
bg_level = np.median(final_img_stacked)
noise = np.std(final_img_stacked)

final_engine = PSFPhotometry(
    psf_model=CircularGaussianPSF(fwhm=5.0),
    fit_shape=(11, 11),
    finder=DAOStarFinder(threshold=1.5 * noise, fwhm=5.0),
    local_bkg_estimator=LocalBackground(inner_radius=15, outer_radius=25),
    aperture_radius=5.0
)
phot_table = final_engine(data=final_img_stacked)

print(phot_table['id', 'x_fit', 'y_fit', 'flux_fit', 'flux_err'])

plt.imshow(final_img_stacked, cmap='gray', vmin=bg_level + (0.5 * noise), vmax=bg_level + (10 * noise))
plt.show()
