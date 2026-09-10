from __future__ import annotations

import glob
import os
import subprocess
from pathlib import Path

import astroalign as aa
import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import sep
from reproject import reproject_interp
from scipy.ndimage import gaussian_filter, shift as ndi_shift
from skimage.registration import phase_cross_correlation
from astropy.coordinates import SkyCoord
from astropy.nddata import NDData
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.wcs import WCS
from photutils.background import LocalBackground
from photutils.detection import DAOStarFinder
from photutils.psf import EPSFBuilder, PSFPhotometry, extract_stars


def _load_image(path):
    return np.asarray(fits.getdata(path), dtype=np.float64)


def build_master_bias(bias_files):
    if not bias_files:
        raise FileNotFoundError("No bias frames were found")
    frames = [_load_image(path) for path in bias_files]
    master = np.median(np.stack(frames), axis=0)
    print(f"Master bias: shape={master.shape}, median={np.median(master):.3f} ADU")
    return master


def build_master_dark(dark_files):
    """Median-combine raw darks that match the science exposure time."""
    if not dark_files:
        raise FileNotFoundError("No dark frames were found")
    frames = [_load_image(path) for path in dark_files]
    master = np.median(np.stack(frames), axis=0)
    print(f"Master dark: shape={master.shape}, median={np.median(master):.3f} ADU")
    return master


def build_master_flat(flat_files, master_bias):
    """Bias-correct and normalize each flat before median combining it."""
    if not flat_files:
        raise FileNotFoundError("No flat frames were found")

    normalized = []
    for path in flat_files:
        data = _load_image(path)
        corrected = data - master_bias
        level = np.nanmedian(corrected)
        if not np.isfinite(level) or level <= 0:
            raise ValueError(f"Flat {path} has invalid calibrated median {level}")
        normalized.append(corrected / level)

    master = np.median(np.stack(normalized), axis=0)
    bad = ~np.isfinite(master) | (master <= 0.05)
    if np.mean(bad) > 0.01:
        print(f"Warning: {100*np.mean(bad):.2f}% of master-flat pixels are invalid or very low")
    master[bad] = np.nan
    return master


def calibrate_science(path, master_dark, master_flat):
    data, header = fits.getdata(path, header=True)
    data = np.asarray(data, dtype=np.float64)
    calibrated = (data - master_dark) / master_flat
    calibrated[~np.isfinite(calibrated)] = np.nan
    return calibrated, header.copy()


def _detection_image(image):
    """Create the native-endian, finite C array expected by SEP/Astroalign."""
    image = np.asarray(image, dtype=np.float64)
    _, median, _ = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    detection = image - median
    finite = np.isfinite(detection)
    if not finite.any():
        raise ValueError("Image has no finite pixels")
    detection[~finite] = 0.0
    return np.require(detection, dtype=np.float32, requirements=["C", "A", "O", "W"])


def _registration_input(image):
    """Return a native, contiguous floating array while preserving calibrated values."""
    output = np.asarray(image, dtype=np.float64)
    return np.require(output, dtype=np.float64, requirements=["C", "A", "O", "W"])


def _alignment_score(image):
    """Robust contrast score used only to choose a reference exposure."""
    _, median, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    if not np.isfinite(std) or std <= 0:
        return -np.inf
    high = np.nanpercentile(image, 99.5)
    return float((high - median) / std)


def _control_points(image, detection_sigma=3.0, min_area=5, maximum=100):
    """Extract bright-source coordinates without Astroalign's exception masking."""
    detection = _detection_image(image)
    try:
        background = sep.Background(detection)
        threshold = detection_sigma * background.globalrms
        sources = sep.extract(
            detection - background.back(),
            threshold,
            minarea=min_area,
        )
    except Exception as exc:
        raise RuntimeError(
            "SEP source extraction failed on a finite, native-endian, contiguous "
            f"array (shape={detection.shape}, dtype={detection.dtype}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if len(sources) < 3:
        raise ValueError(
            f"Only {len(sources)} sources were detected; at least 3 are required. "
            "Try DETECTION_SIGMA=2.5 if the frame visibly contains several stars."
        )
    order = np.argsort(sources["flux"])[::-1][:maximum]
    return np.column_stack((sources["x"][order], sources["y"][order]))


def solve_reference_wcs(image, header, filter_name):
    """Plate-solve one reference frame through the local WSL Astrometry.net install."""
    if "RA" not in header or "DEC" not in header:
        raise KeyError("Reference FITS header must contain RA and DEC")

    stem = f"temp_ref_{filter_name}"
    input_path = Path(f"{stem}.fits")
    solved_path = Path(f"{stem}.new")
    fits.PrimaryHDU(data=image, header=header).writeto(input_path, overwrite=True)

    coord = SkyCoord(f"{header['RA']} {header['DEC']}", unit=(u.hourangle, u.deg))
    command = [
        "wsl", "solve-field", str(input_path),
        "--ra", str(coord.ra.deg), "--dec", str(coord.dec.deg),
        "--radius", "2.0", "--downsample", "2", "--overwrite", "--no-plots",
    ]
    try:
        result = subprocess.run(command, check=False)
        if result.returncode != 0 or not solved_path.exists():
            raise RuntimeError(
                f"Astrometry.net did not create {solved_path}; return code={result.returncode}"
            )
        return WCS(fits.getheader(solved_path))
    finally:
        for path in glob.glob(f"{stem}*"):
            try:
                os.remove(path)
            except OSError:
                pass


def align_to_reference(image, reference, detection_sigma=3.0):
    """Find one transform on detection images and apply it to calibrated pixels."""
    source_points = _control_points(image, detection_sigma=detection_sigma)
    target_points = _control_points(reference, detection_sigma=detection_sigma)
    transform, matched = aa.find_transform(
        source_points,
        target_points,
        max_control_points=100,
    )
    aligned, footprint = aa.apply_transform(
        transform,
        _registration_input(image),
        _registration_input(reference),
        fill_value=np.nan,
        propagate_mask=False,
    )
    aligned = np.asarray(aligned, dtype=np.float64)
    aligned[np.asarray(footprint, dtype=bool)] = np.nan
    return aligned, transform, matched


def align_translation_to_reference(image, reference, maximum_shift_fraction=0.25):
    """Fallback alignment for tracking drift when star-triangle matching fails."""
    source = _detection_image(image)
    target = _detection_image(reference)

    source = gaussian_filter(np.maximum(source, 0.0), sigma=1.25)
    target = gaussian_filter(np.maximum(target, 0.0), sigma=1.25)
    window = np.outer(np.hanning(target.shape[0]), np.hanning(target.shape[1])).astype(
        np.float32
    )
    shift_yx, error, phasediff = phase_cross_correlation(
        target * window,
        source * window,
        upsample_factor=10,
        normalization=None,
    )
    if not np.all(np.isfinite(shift_yx)):
        raise RuntimeError(f"Phase correlation returned a non-finite shift: {shift_yx}")

    limits = maximum_shift_fraction * np.asarray(target.shape)
    if np.any(np.abs(shift_yx) > limits):
        raise RuntimeError(
            f"Phase-correlation shift {shift_yx} exceeds safety limits {limits} pixels"
        )

    science = _registration_input(image)
    finite = np.isfinite(science)
    fill = float(np.nanmedian(science))
    shifted = ndi_shift(
        np.where(finite, science, fill),
        shift=shift_yx,
        order=3,
        mode="constant",
        cval=fill,
        prefilter=True,
    )
    valid = ndi_shift(
        finite.astype(np.uint8),
        shift=shift_yx,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(bool)
    shifted[~valid] = np.nan
    return shifted, shift_yx, float(error), float(phasediff)


def reproject_to_reference(image, source_wcs, target_wcs, target_shape):
    """Put a solved filter stack on the target filter's celestial pixel grid."""
    reprojected, footprint = reproject_interp(
        (image, source_wcs),
        target_wcs,
        shape_out=target_shape,
        order="bicubic",
        return_footprint=True,
    )
    reprojected = np.asarray(reprojected, dtype=np.float64)
    reprojected[np.asarray(footprint) <= 0] = np.nan
    return reprojected, footprint


def process_channel(
    filter_name,
    raw_files,
    flat_files,
    master_bias,
    master_dark,
    solve_wcs=False,
    detection_sigma=3.0,
):
    """Calibrate, register, and sigma-clipped-stack one filter."""
    print("\n==========================================")
    print(f" PROCESSING FILTER: {filter_name}")
    print("==========================================")
    if not raw_files:
        raise FileNotFoundError(f"No science files found for {filter_name}")

    master_flat = build_master_flat(flat_files, master_bias)
    calibrated = []
    headers = []
    for path in raw_files:
        image, header = calibrate_science(path, master_dark, master_flat)
        calibrated.append(image)
        headers.append(header)

    scores = [_alignment_score(image) for image in calibrated]
    source_counts = []
    for image in calibrated:
        try:
            source_counts.append(
                len(
                    _control_points(
                        image,
                        detection_sigma=max(4.0, detection_sigma),
                        min_area=5,
                        maximum=200,
                    )
                )
            )
        except Exception:
            source_counts.append(0)
    reference_index = max(
        range(len(calibrated)),
        key=lambda index: (source_counts[index], scores[index]),
    )
    reference = calibrated[reference_index]
    reference_header = headers[reference_index]
    print(
        f"Reference: frame {reference_index + 1}/{len(calibrated)} "
        f"({Path(raw_files[reference_index]).name}), "
        f"sources={source_counts[reference_index]}, score={scores[reference_index]:.2f}"
    )

    reference_wcs = None
    if solve_wcs:
        print(f"Plate-solving the {filter_name} reference with Astrometry.net...")
        reference_wcs = solve_reference_wcs(reference, reference_header, filter_name)
        print("Plate solve succeeded.")

    aligned = [reference.copy()]
    failures = []
    astroalign_count = 1
    phase_count = 0
    for index, image in enumerate(calibrated):
        if index == reference_index:
            continue
        print(f"Aligning {filter_name} frame {index + 1}/{len(calibrated)}...")
        try:
            registered, transform, matched = align_to_reference(
                image, reference, detection_sigma=detection_sigma
            )
            aligned.append(registered)
            print(
                f"  OK: matched={len(matched[0])}, rotation={np.degrees(transform.rotation):+.3f} deg, "
                f"scale={transform.scale:.6f}, shift=({transform.translation[0]:+.2f}, "
                f"{transform.translation[1]:+.2f}) px"
            )
            astroalign_count += 1
        except Exception as astroalign_exc:
            print(
                f"  Astroalign unavailable for this frame: "
                f"{type(astroalign_exc).__name__}: {astroalign_exc}"
            )
            try:
                registered, shift_yx, error, _ = align_translation_to_reference(
                    image, reference
                )
                aligned.append(registered)
                phase_count += 1
                print(
                    f"  PHASE FALLBACK OK: shift=(x={shift_yx[1]:+.2f}, "
                    f"y={shift_yx[0]:+.2f}) px, error={error:.6f}"
                )
            except Exception as phase_exc:
                message = (
                    f"Astroalign={type(astroalign_exc).__name__}: {astroalign_exc}; "
                    f"phase={type(phase_exc).__name__}: {phase_exc}"
                )
                failures.append((Path(raw_files[index]).name, message))
                print(f"  SKIPPED: {message}")

    print(
        f"{filter_name}: aligned {len(aligned)}/{len(calibrated)} frames "
        f"(Astroalign={astroalign_count}, phase fallback={phase_count})"
    )
    if failures:
        print(f"{filter_name}: skipped {len(failures)} frames")
    if len(aligned) < 2:
        raise RuntimeError(
            f"Only the {filter_name} reference frame survived; no meaningful stack can be made. "
            "The first reported alignment exception is the key diagnostic."
        )

    cube = np.stack(aligned).astype(np.float64)
    clipped = sigma_clip(
        cube,
        sigma=3.0,
        maxiters=5,
        cenfunc="median",
        stdfunc="mad_std",
        axis=0,
        masked=True,
    )
    stack = np.ma.mean(clipped, axis=0).filled(np.nan)
    return stack, reference_wcs, reference_header, master_flat, failures


def normalize_channel(image, p_min=1.0, p_max=99.7, asinh_strength=8.0):
    """Percentile clip and apply a gentle asinh stretch for RGB display."""
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=float)
    low, high = np.nanpercentile(image, (p_min, p_max))
    if not np.isfinite(high) or high <= low:
        return np.zeros_like(image, dtype=float)
    scaled = np.clip((image - low) / (high - low), 0.0, 1.0)
    stretched = np.arcsinh(asinh_strength * scaled) / np.arcsinh(asinh_strength)
    return np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0)


def run_psf_photometry(image_data):
    """Optional PSF photometry; this is not required to make the RGB stack."""
    _, median, noise = sigma_clipped_stats(image_data, sigma=3.0, maxiters=5)
    detection = np.nan_to_num(image_data - median, nan=0.0)
    finder = DAOStarFinder(threshold=5.0 * noise, fwhm=5.0)
    catalog = finder(detection)
    if catalog is None or len(catalog) < 5:
        print("Warning: fewer than five suitable sources were found for ePSF construction.")
        return None

    height, width = image_data.shape
    margin = 15
    valid = catalog[
        (catalog["xcentroid"] > margin)
        & (catalog["xcentroid"] < width - margin)
        & (catalog["ycentroid"] > margin)
        & (catalog["ycentroid"] < height - margin)
    ]
    if len(valid) < 5:
        print("Warning: fewer than five detected stars remain after edge filtering.")
        return None

    positions = valid["xcentroid", "ycentroid"]
    positions.rename_column("xcentroid", "x")
    positions.rename_column("ycentroid", "y")
    stars = extract_stars(NDData(data=image_data), positions, size=15)
    epsf, _ = EPSFBuilder(maxiters=10, progress_bar=False)(stars)
    engine = PSFPhotometry(
        psf_model=epsf,
        fit_shape=(11, 11),
        finder=DAOStarFinder(threshold=5.0 * noise, fwhm=5.0),
        local_bkg_estimator=LocalBackground(inner_radius=15, outer_radius=25),
        aperture_radius=5.0,
    )
    return engine(data=image_data)