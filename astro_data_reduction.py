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


# ============================================================
# USER CONFIGURATION
# Change this section when processing a different object.
# I have included examples of what mine have looked like, but you need to change it according to your object and file location.
# Also, you can change Run_PSF_Photometry to False if you are not interested in data, just creating an image.
# ============================================================

OBJECT_NAME = "NGC253"

DATA_ROOT = Path(r"D:\SARA Data\101025CT")
CALIBRATION_DIR = DATA_ROOT / "NGC253"
SCIENCE_DIR = DATA_ROOT / "NGC253"
OUTPUT_DIR = SCIENCE_DIR / f"{OBJECT_NAME}_reduced_rgb"

FILTERS = ("R", "G", "B")

BIAS_PATTERN = "NGC253_None_bias_*.fits"
DARK_PATTERN = "NGC253_None_dark_*.fits"

SCIENCE_PATTERNS = {
    "R": "NGC253_Bessell_R_*.fits",
    "G": "NGC253_Bessell_V_*.fits",
    "B": "NGC253_Bessell_B_*.fits",
}

FLAT_PATTERNS = {
    "R": "NGC23_Bessell_R_flat_*.fits",
    "G": "NGC253_Bessell_V_flat_*.fits",
    "B": "NGC253_Bessell_B_flat_*.fits",
}

RUN_PSF_PHOTOMETRY = True
DETECTION_SIGMA = 3.0


# ============================================================
# PIPELINE
# Nothing below this point normally needs to be renamed.
# ============================================================

def sorted_glob(pattern):
    return sorted(glob.glob(str(pattern)))


def output_header(reference_header, reference_wcs):
    header = reference_header.copy()
    if reference_wcs is not None:
        header.update(reference_wcs.to_header(relax=True))
    return header


def main():
    bias_files = sorted_glob(CALIBRATION_DIR / BIAS_PATTERN)
    dark_files = sorted_glob(CALIBRATION_DIR / DARK_PATTERN)

    science_files = {
        channel: sorted_glob(SCIENCE_DIR / SCIENCE_PATTERNS[channel])
        for channel in FILTERS
    }

    flat_files = {
        channel: sorted_glob(CALIBRATION_DIR / FLAT_PATTERNS[channel])
        for channel in FILTERS
    }

    print(f"Object: {OBJECT_NAME}")
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
        stack, wcs_solution, reference_header, master_flat, failures = (
            process_channel(
                filter_name=channel,
                raw_files=science_files[channel],
                flat_files=flat_files[channel],
                master_bias=master_bias,
                master_dark=master_dark,
                solve_wcs=True,
                detection_sigma=DETECTION_SIGMA,
            )
        )

        channel_stacks[channel] = stack
        channel_flats[channel] = master_flat
        channel_wcs[channel] = wcs_solution

        if channel == "R":
            r_header = reference_header

        if failures:
            print(f"{channel}: {len(failures)} frame(s) were skipped.")

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
            f"valid footprint={100 * np.mean(footprint > 0):.2f}%"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = output_header(r_header, channel_wcs["R"])

    for channel in FILTERS:
        fits.PrimaryHDU(
            aligned[channel],
            header=header,
        ).writeto(
            OUTPUT_DIR
            / f"{OBJECT_NAME}_{channel}_stack_aligned_to_R.fits",
            overwrite=True,
        )

        fits.PrimaryHDU(
            channel_flats[channel]
        ).writeto(
            OUTPUT_DIR / f"{OBJECT_NAME}_master_flat_{channel}.fits",
            overwrite=True,
        )

    fits.PrimaryHDU(master_bias).writeto(
        OUTPUT_DIR / f"{OBJECT_NAME}_master_bias.fits",
        overwrite=True,
    )

    fits.PrimaryHDU(master_dark).writeto(
        OUTPUT_DIR / f"{OBJECT_NAME}_master_dark.fits",
        overwrite=True,
    )

    if RUN_PSF_PHOTOMETRY:
        table = run_psf_photometry(aligned["G"])

        if table is not None:
            table.write(
                OUTPUT_DIR
                / f"{OBJECT_NAME}_G_channel_psf_photometry.ecsv",
                overwrite=True,
            )
            table.pprint(max_lines=20, max_width=150)
            print(f"PSF photometry: {len(table)} sources")

    rgb = np.dstack(
        [
            normalize_channel(aligned["R"]),
            normalize_channel(aligned["G"]),
            normalize_channel(aligned["B"]),
        ]
    )

    rgb_path = OUTPUT_DIR / f"{OBJECT_NAME}_RGB.png"
    plt.imsave(rgb_path, rgb, origin="lower")

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb, origin="lower")
    plt.title(f"{OBJECT_NAME} BVR Composite")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"\nFinished. Products saved in: {OUTPUT_DIR}")
    print(f"RGB image: {rgb_path}")


if __name__ == "__main__":
    main()
