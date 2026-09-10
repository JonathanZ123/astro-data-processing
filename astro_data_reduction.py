from pathlib import Path
import glob

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np

from pipeline_utils import (
    build_master_bias,
    build_master_dark,
    normalize_channel,
    process_channel,
    reproject_to_reference,
    run_psf_photometry,
)

DATA_ROOT = Path(r"D:\main\path")
CALIBRATION_DIR = DATA_ROOT / "calibration_path"
SCIENCE_DIR = DATA_ROOT / "science_path"
OUTPUT_DIR = SCIENCE_DIR / "reduced_rgb"

FILTERS = ("R", "G", "B")
RUN_PSF_PHOTOMETRY = False
DETECTION_SIGMA = 3.0


def sorted_glob(pattern):
    return sorted(glob.glob(str(pattern)))


def output_header(reference_header, reference_wcs):
    header = reference_header.copy()
    if reference_wcs is not None:
        header.update(reference_wcs.to_header(relax=True))
    return header


def main():
    bias_files = sorted_glob(CALIBRATION_DIR / "3c454_None__bias_*.fits")
    dark_files = sorted_glob(CALIBRATION_DIR / "3c454_None__dark_*.fits")
    science_files = {
        "R": sorted_glob(SCIENCE_DIR / "NGC253_Bessell_R_*.fits"),
        "G": sorted_glob(SCIENCE_DIR / "NGC253_Bessell_V_*.fits"),
        "B": sorted_glob(SCIENCE_DIR / "NGC253_Bessell_B_*.fits"),
    }
    flat_files = {
        "R": sorted_glob(CALIBRATION_DIR / "3c454_Bessell_R__flat_*.fits"),
        "G": sorted_glob(CALIBRATION_DIR / "3c454_Bessell_V__flat_*.fits"),
        "B": sorted_glob(CALIBRATION_DIR / "3c454_Bessell_B__flat_*.fits"),
    }

    print(f"Bias files: {len(bias_files)}")
    print(f"Dark files: {len(dark_files)}")
    for channel in FILTERS:
        print(
            f"{channel}: flats={len(flat_files[channel])}, "
            f"science={len(science_files[channel])}"
        )

    master_bias = build_master_bias(bias_files)
    master_dark = build_master_dark(dark_files)

    channel_stacks = {}
    channel_flats = {}
    channel_wcs = {}
    r_header = None

    for channel in FILTERS:
        stack, wcs_solution, reference_header, master_flat, failures = process_channel(
            filter_name=channel,
            raw_files=science_files[channel],
            flat_files=flat_files[channel],
            master_bias=master_bias,
            master_dark=master_dark,
            solve_wcs=True,
            detection_sigma=DETECTION_SIGMA,
        )
        channel_stacks[channel] = stack
        channel_flats[channel] = master_flat
        channel_wcs[channel] = wcs_solution
        if channel == "R":
            r_header = reference_header

    print("\n==========================================")
    print(" REPROJECTING SOLVED G AND B STACKS TO THE R WCS GRID")
    print("==========================================")
    aligned = {"R": channel_stacks["R"]}
    for channel in ("G", "B"):
        registered, footprint = reproject_to_reference(
            channel_stacks[channel],
            channel_wcs[channel],
            channel_wcs["R"],
            channel_stacks["R"].shape,
        )
        aligned[channel] = registered
        print(
            f"{channel}->R: WCS reprojection complete; "
            f"valid footprint={100*np.mean(footprint > 0):.2f}%"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = output_header(r_header, channel_wcs["R"])
    for channel in FILTERS:
        fits.PrimaryHDU(aligned[channel], header=header).writeto(
            OUTPUT_DIR / f"NGC253_{channel}_stack_aligned_to_R.fits",
            overwrite=True,
        )
        fits.PrimaryHDU(channel_flats[channel]).writeto(
            OUTPUT_DIR / f"master_flat_{channel}.fits",
            overwrite=True,
        )

    fits.PrimaryHDU(master_bias).writeto(OUTPUT_DIR / "master_bias.fits", overwrite=True)
    fits.PrimaryHDU(master_dark).writeto(OUTPUT_DIR / "master_dark.fits", overwrite=True)

    if RUN_PSF_PHOTOMETRY:
        table = run_psf_photometry(aligned["G"])
        if table is not None:
            table.write(OUTPUT_DIR / "G_channel_psf_photometry.ecsv", overwrite=True)
            print(f"PSF photometry: {len(table)} sources")

    rgb = np.dstack(
        [
            normalize_channel(aligned["R"]),
            normalize_channel(aligned["G"]),
            normalize_channel(aligned["B"]),
        ]
    )
    print('========================================================')
    table.pprint(max_lines=20, max_width=150)
    plt.imsave(OUTPUT_DIR / "NGC253_RGB.png", rgb, origin="lower")
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb, origin="lower")
    plt.show()
    print(f"\nFinished. Products saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
