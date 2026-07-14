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
API_KEY = os.getenv("ASTROMETRY_API_KEY")

dark_files = glob.glob(r"D:\SARA Data\112725RM\3c273\Dark30s_Empty_*.fits")
bias_files = glob.glob(r"D:\SARA Data\112725RM\3c273\Bias_Empty_*.fits")
flat_files = glob.glob(r"D:\SARA Data\112725RM\3c273\sarm20251111_flat_JohnsonV_*.fits")
raw_science_files = glob.glob(r"D:\SARA Data\112725RM\3c273\3c273_Johnson_V_*_light.fits")

print("Dark Files Found:", len(dark_files))
print("Bias Files Found:", len(bias_files))
print("Flat Files Found:", len(flat_files))
print("Raw Science Files Found:", len(raw_science_files))
print("--------------------------------")

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

master_flat_raw = np.median(flat_cube, axis=0)
master_flat = master_flat_raw - master_bias
master_flat = master_flat / np.median(master_flat)
master_flat[master_flat == 0] = 1

calibrated_science_list = []
raw_science_headers = []

for raw_file in raw_science_files:
    hdu = fits.open(raw_file)[0]
    raw_science_headers.append(hdu.header)
    clean_frame = (hdu.data - master_dark) / master_flat
    calibrated_science_list.append(clean_frame)

ast = AstrometryNet()
ast.api_key = API_KEY

ref_data = calibrated_science_list[0]
ref_header = raw_science_headers[0]

temp_ref = "temp_ref.fits"
hdu_ref = fits.PrimaryHDU(data=ref_data, header=ref_header)
hdu_ref.writeto(temp_ref, overwrite=True)

ref_wcs = None

try:
    ra_hint = ref_header.get('RA')
    dec_hint = ref_header.get('DEC')
    
    if ra_hint and dec_hint:
        coord = SkyCoord(f"{ra_hint} {dec_hint}", unit=(u.hourangle, u.deg))
        ra_float = coord.ra.deg
        dec_float = coord.dec.deg

        ref_wcs = ast.solve_from_image(temp_ref, 
                                       center_ra=ra_float, 
                                       center_dec=dec_float, 
                                       radius=2.0, 
                                       solve_timeout=300)
    else:
        ref_wcs = ast.solve_from_image(temp_ref, solve_timeout=300)
except Exception as e:
    print(f"An error occurred during reference solve: {e}")
    print("The server dropped the connection. Retrying with coordinates hint...")
    try:
        ref_wcs = ast.solve_from_image(temp_ref, 
                                       center_ra=ra_float, 
                                       center_dec=dec_float, 
                                       radius=2.0, 
                                       solve_timeout=150)
    except:
        ref_wcs = None

if os.path.exists(temp_ref):
    try:
        os.remove(temp_ref)
    except Exception as file_err:
        print(f"Temporary file cleanup warning: {file_err} (You can ignore this)")

if ref_wcs is None:
    raise RuntimeError("Could not plate-solve the reference frame due to network drops. Pipeline stopped.")

aligned_science_list = [ref_data]

for i in range(1, len(calibrated_science_list)):
    current_data = calibrated_science_list[i]
    current_header = raw_science_headers[i]
    
    temp_current = f"temp_frame_{i}.fits"
    hdu_curr = fits.PrimaryHDU(data=current_data, header=current_header)
    hdu_curr.writeto(temp_current, overwrite=True)
    
    print(f"Solving frame {i+1} of {len(calibrated_science_list)}...")
    
    current_ra = current_header.get('RA')
    current_dec = current_header.get('DEC')
    
    try:
        if current_ra and current_dec:
            coord = SkyCoord(f"{current_ra} {current_dec}", unit=(u.hourangle, u.deg))
            current_wcs = ast.solve_from_image(temp_current, 
                                               center_ra=coord.ra.deg, 
                                               center_dec=coord.dec.deg, 
                                               radius=2.0, 
                                               solve_timeout=300)
        else:
            current_wcs = ast.solve_from_image(temp_current, solve_timeout=300)
            
        if current_wcs is not None:
            print(f"Aligning frame {i+1} to reference grid...")
            aligned_array, footprint = reproject_interp(
                (current_data, current_wcs), 
                output_projection=ref_wcs, 
                shape_out=ref_data.shape
            )
            aligned_science_list.append(aligned_array)
        else:
            print(f"Warning: Skipping frame {i+1} because it couldn't be plate-solved.")

    except Exception as e:
        print(f"Warning: Error processing frame {i+1}: {e}. Skipping.")

    if os.path.exists(temp_current):
        try:
            os.remove(temp_current)
        except:
            pass

science_cube = np.array(aligned_science_list)
final_img_stacked = np.median(science_cube, axis=0)

bg_level = np.median(final_img_stacked)
noise = np.std(final_img_stacked)

print("--------------------------------")
print("Typical Background Level:", bg_level)
print("Typical Noise Level:", noise)

auto_vmin = bg_level + (0.5 * noise)
auto_vmax = bg_level + (10 * noise)

finder_tool = DAOStarFinder(threshold=1.5 * noise, fwhm=5.0)
gaussian_model = CircularGaussianPSF(fwhm=5.0)
bg_tool = LocalBackground(inner_radius=15, outer_radius=25)

final_engine = PSFPhotometry(
    psf_model=gaussian_model,
    fit_shape=(11, 11),
    finder=finder_tool,
    local_bkg_estimator=bg_tool,
    aperture_radius=5.0
)

phot_table = final_engine(data=final_img_stacked)

print("--------------------------------")
print("Found", len(phot_table), "total sources.")
print("--------------------------------")

print(phot_table['id', 'x_fit', 'y_fit', 'flux_fit', 'flux_err'])

plt.imshow(final_img_stacked, cmap='gray', vmin=auto_vmin, vmax=auto_vmax)
plt.show()
