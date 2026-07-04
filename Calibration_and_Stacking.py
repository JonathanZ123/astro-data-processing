import glob
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt

dark_files = glob.glob(r"D:\SARA Data\112725RM\3c273\Dark30s_Empty_*.fits")
bias_files = glob.glob(r"D:\SARA Data\112725RM\3c273\Bias_Empty_*.fits")
flat_files = glob.glob(r"D:\SARA Data\112725RM\3c273\sarm20251111_flat_JohnsonV_*.fits")
raw_science_files = glob.glob(r"D:\SARA Data\112725RM\3c273\3c273_Johnson_V_*_light.fits")

print("Dark Files Found:",(len(dark_files)))
print("bias Files Found:",(len(bias_files)))
print("flat Files Found:",(len(flat_files)))
print("Raw Science Files Found:",(len(raw_science_files)))
print("--------------------------------")

dark_list = []
bias_list = []
flat_list = []
raw_science_list = []

for dark_file in dark_files:
    data = fits.open(dark_file)[0].data
    dark_list.append(data)
for bias_file in bias_files:
    data = fits.open(bias_file)[0].data
    bias_list.append(data)
for flat_file in flat_files:
    data = fits.open(flat_file)[0].data
    flat_list.append(data)
for raw_file in raw_science_files:
    data = fits.open(raw_file)[0].data
    raw_science_list.append(data)

dark_cube = np.array(dark_list)
bias_cube = np.array(bias_list)
flat_cube = np.array(flat_list)
science_cube = np.array(raw_science_list)

final_img_stacked = np.mean(science_cube, axis=0)
master_dark = np.median(dark_cube, axis=0)
master_bias = np.median(bias_cube, axis=0)
master_flat_raw = np.median(flat_cube, axis=0)
master_flat = master_flat_raw - master_bias
master_flat = master_flat / np.median(master_flat)
master_flat[master_flat == 0] = 1

calibrated_img = (final_img_stacked - master_dark) / master_flat

bg_level = np.median(calibrated_img)
noise = np.std(calibrated_img)

print("Typical Background Level:",bg_level)
print("Typical Noise Level:",noise)
print("--------------------------------")

auto_vmin = bg_level - (1 * noise)
auto_vmax = bg_level + (5 * noise)

plt.imshow(calibrated_img, cmap = 'gray', vmin = auto_vmin, vmax = auto_vmax)
plt.show()