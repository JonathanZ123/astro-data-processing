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

# Fetch list of file paths for calibration and science frames
dark_files = glob.glob(r"D:\your\file\here_*.fits")
bias_files = glob.glob(r"D:\your\file\here_*.fits")
flat_files = glob.glob(r"D:\your\file\here_*.fits")
raw_science_files = glob.glob(r"D:\your\file\here_*.fits")

# Output counts of located calibration and target frames
print("Dark Files Found:", len(dark_files))
print("Bias Files Found:", len(bias_files))
print("Flat Files Found:", len(flat_files))
print("Raw Science Files Found:", len(raw_science_files))
print("--------------------------------")

# Extract raw 2D pixel matrices from primary HDUs
dark_list = [fits.open(f)[0].data for f in dark_files]
bias_list = [fits.open(f)[0].data for f in bias_files]
flat_list = [fits.open(f)[0].data for f in flat_files]

# Combine calibration frames into master calibration images
master_dark = np.median(np.array(dark_list), axis=0)  # Median-combine dark frames to remove thermal noise.
master_bias = np.median(np.array(bias_list), axis=0)  # Median-combine bias frames to remove readout noise.

master_flat = np.median(np.array(flat_list), axis=0) - master_bias  # Subtract master bias from raw flats.
master_flat = master_flat / np.median(master_flat)  # Normalize master flat around a median of 1.0.
master_flat[master_flat == 0] = 1  # Prevent divide-by-zero errors by setting zero pixels to 1.

calibrated_science_list = []  # Storage list for calibrated science pixel arrays.
raw_science_headers = []      # Storage list for corresponding FITS headers.

for file in raw_science_files:  # Loop through each raw science frame path.
    hdu = fits.open(file)[0]  # Open FITS file and extract primary HDU object.
    raw_science_headers.append(hdu.header)  # Save frame header metadata.
    calibrated_science_list.append((hdu.data - master_dark) / master_flat)  # Calibrate science data: (Raw - Dark) / Flat.

# ==========================================
# Solving Reference Frame
# ==========================================
ref_data = calibrated_science_list[0]  # Select first calibrated image as reference alignment frame.
ref_header = raw_science_headers[0]    # Grab matching reference FITS header.

# Write out temporary file to disk for WSL solver
hdu_ref = fits.PrimaryHDU(data=ref_data, header=ref_header)  # Package reference array and header into primary HDU.
hdu_ref.writeto("temp_ref.fits", overwrite=True)             # Save temporary FITS image to disk.

# Parse coordinates for search hints
coord = SkyCoord(f"{ref_header['RA']} {ref_header['DEC']}", unit=(u.hourangle, u.deg))  # Convert RA/Dec string to SkyCoord.
ra_deg = coord.ra.deg    # Extract Right Ascension in decimal degrees.
dec_deg = coord.dec.deg  # Extract Declination in decimal degrees.

# Build command line solver string
command_ref = (
    f"wsl solve-field temp_ref.fits "               # Call Linux solver via WSL on reference image.
    f"--ra {ra_deg} --dec {dec_deg} --radius 2.0 "  # Target solution search within 2 degrees of header coordinates.
    f"--overwrite --no-plots"                       # Overwrite existing outputs and suppress plot generation.
)

print("Solving reference frame via WSL...")  # Status message.
subprocess.run(command_ref, shell=True)      # Run plate-solver in system shell.

# Load newly created WCS header from solver output
if os.path.exists("temp_ref.new"):  # Check if plate solver generated a output file.
    ref_wcs = WCS(fits.getheader("temp_ref.new"))  # Load solved WCS coordinate transformation from output header.
else:
    raise RuntimeError("Failed to plate-solve the reference frame.")  # Halt program if reference frame fails to solve.

# Clean up all temporary files created by reference solver
for temp_file in glob.glob("temp_ref*"):  # Search for all generated temporary reference files.
    if os.path.exists(temp_file):         # Check file existence.
        os.remove(temp_file)              # Delete temporary file.

# ==========================================
# Alignment Loop
# ==========================================
aligned_science_list = [ref_data]  # Initialize the aligned stack with unshifted reference image.

for i in range(1, len(calibrated_science_list)):    # Loop over remaining science frames.
    current_data = calibrated_science_list[i]       # Get current frame pixel data.
    current_header = raw_science_headers[i]         # Get current frame header metadata.
    
    temp_current = f"temp_frame_{i}.fits"  # Define unique temporary file name.
    hdu_curr = fits.PrimaryHDU(data=current_data, header=current_header)  # Package current data into HDU object.
    hdu_curr.writeto(temp_current, overwrite=True)  # Write temporary frame to disk.
    
    print(f"\nSolving frame {i+1} of {len(calibrated_science_list)}")  # Print loop progress.
    
    coord_curr = SkyCoord(f"{current_header['RA']} {current_header['DEC']}", unit=(u.hourangle, u.deg))  # Parse frame sky coordinates.
    
    # Build solver command for current loop iteration
    command_loop = (
        f"wsl solve-field {temp_current} "  # Call WSL plate-solver on temporary file.
        f"--ra {coord_curr.ra.deg} --dec {coord_curr.dec.deg} --radius 1.0 "  # Search within 1 degree of header coordinates.
        f"--overwrite --no-plots"  # Overwrite previous runs and disable plot generation.
    )
    subprocess.run(command_loop, shell=True)  # Run solver process.
    
    solved_filename = temp_current.replace(".fits", ".new")  # Predict solved output file name.
    
    if os.path.exists(solved_filename):        # Check if current frame plate-solved successfully.
        current_wcs = WCS(fits.getheader(solved_filename))  # Parse solved WCS from output header.
        
        # Reproject/align current image matrix to match reference frame WCS
        aligned_array, _ = reproject_interp(
            (current_data, current_wcs),  # Source image array and source WCS coordinate system.
            output_projection=ref_wcs,    # Target WCS coordinate system (reference frame).
            shape_out=ref_data.shape      # Target output image dimensions.
        )
        aligned_science_list.append(aligned_array)  # Store aligned array into image stack list.
    else:
        print(f"Warning: Frame {i+1} failed to solve locally. Skipping.")  # Warn if frame failed to solve.
        
    # Clean up temporary frame files generated in this iteration
    for temp_file in glob.glob(f"temp_frame_{i}*"):  # Match all temporary files for iteration i.
        if os.path.exists(temp_file):                # Verify file existence.
            os.remove(temp_file)                     # Delete file.

# --- Stacking with Sigma Clipping and NaN Protections ---
aligned_cube = np.array(aligned_science_list, dtype=float)  # Combine aligned 2D arrays into a 3D data cube.
aligned_cube[~np.isfinite(aligned_cube)] = np.nan           # Replace infinity or invalid pixels with NaN values.

clipped_cube = sigma_clip(
    aligned_cube,          # Input 3D data stack.
    sigma=3.0,             # Mask pixel values exceeding n standard deviations from median.
    maxiters=5,            # Run up to n clipping iterations.
    cenfunc='median',      # Use median as central estimator.
    stdfunc='std',         # Use standard deviation as spread estimator.
    axis=0,                # Perform rejection along image stack axis.
    masked=True            # Return output array as a masked array.
)

final_img_stacked = np.ma.mean(clipped_cube, axis=0).filled(np.nan)  # Average unmasked pixels across stack, filling masked areas with NaN.

bg_level = np.nanmedian(final_img_stacked)  # Calculate median background level of stacked image.
noise = np.nanstd(final_img_stacked)        # Calculate background standard deviation.

final_img_stacked = np.nan_to_num(final_img_stacked, nan=bg_level)  # Fill missing edge NaNs with background level.

print("--------------------------------")
print("Typical Background Level:", bg_level)
print("Typical Noise Level:", noise)

# --- ePSF Photometry Engine ---
# 1. High-threshold detection pass for bright PSF candidate stars
psf_finder = DAOStarFinder(threshold=5.0 * noise, fwhm=5.0)  # Configure star finder for bright sources (5-sigma threshold).
star_catalog = psf_finder(final_img_stacked)  # Detect bright candidate stars across stacked image.

if star_catalog is not None and len(star_catalog) > 0:  # Proceed if bright stars were found.
    nddata = NDData(data=final_img_stacked)  # Package stacked image into NDData object.
    
    # Filter candidate stars too close to edges for complete cutouts
    edge_margin = 15                # Minimum distance in pixels required from image border.
    h, w = final_img_stacked.shape  # Read stacked image height and width.
    valid_stars = star_catalog[
        (star_catalog['xcentroid'] > edge_margin) &        # Exclude stars near left boundary.
        (star_catalog['xcentroid'] < w - edge_margin) &    # Exclude stars near right boundary.
        (star_catalog['ycentroid'] > edge_margin) &        # Exclude stars near bottom boundary.
        (star_catalog['ycentroid'] < h - edge_margin)      # Exclude stars near top boundary.
    ]
    
    stars_tbl = valid_stars['xcentroid', 'ycentroid']  # Extract coordinate columns for star building.
    stars_tbl.rename_column('xcentroid', 'x')  # Rename x centroid column to match required builder format.
    stars_tbl.rename_column('ycentroid', 'y')  # Rename y centroid column to match required builder format.

    # 2. Extract cutouts and build empirical PSF model
    stars = extract_stars(nddata, stars_tbl, size=15)  # Extract 15x15 pixel cutout boxes centered on candidate stars.
    epsf_builder = EPSFBuilder(shape=None, maxiters=10, progress_bar=False)  # Configure ePSF construction algorithm.
    epsf, fitted_stars = epsf_builder(stars)  # Build effective empirical PSF model from extracted star profiles.

    # 3. Perform profile-fitting photometry using built ePSF
    final_engine = PSFPhotometry(
        psf_model=epsf,      # Pass constructed empirical PSF model.
        fit_shape=(11, 11),  # Set fitting box sub-region size (11x11 pixels).
        finder=DAOStarFinder(threshold=1.5 * noise, fwhm=5.0),  # Detect faint sources down to 1.5-sigma threshold.
        local_bkg_estimator=LocalBackground(inner_radius=15, outer_radius=25),  # Subtract local background sky annulus.
        aperture_radius=5.0  # Initial guessing aperture radius.
    )

    phot_table = final_engine(data=final_img_stacked)  # Run profile-fitting PSF photometry on final stacked image.
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

# --- Visualization ---
plt.imshow(final_img_stacked, cmap='gray', vmin=bg_level + (0.5 * noise), vmax=bg_level + (10 * noise))
plt.show()
