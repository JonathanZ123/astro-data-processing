# Astronomical Data Reduction Pipeline

A Python-based astronomical data reduction pipeline designed to calibrate raw science frames, align and stack images, plate-solve sky coordinates, and perform precision empirical Point Spread Function (ePSF) photometry.

---

## Image Calibration Formula

Raw science images are calibrated frame-by-frame using standard CCD calibration equations:

$$\text{Calibrated Image} = \frac{\text{Raw Science} - \text{Master Dark}}{\left( \frac{\text{Raw Flat} - \text{Master Bias}}{\text{Normalization Factor}} \right)}$$

---

## Features

* **Calibration:** Automated creation of Master Dark, Master Bias, and normalized Master Flat frames.
* **Astrometry:** Local plate-solving using `Astrometry.net` via Windows Subsystem for Linux (WSL).
* **Alignment & Stacking:** Multi-frame image registration via `reproject` and robust outlier rejection using sigma clipping (`astropy.stats.sigma_clip`).
* **ePSF Photometry:** Empirical PSF model construction using `photutils.psf.EPSFBuilder` and star extraction for accurate flux measurement.

---

## Prerequisites & Installation

### 1. Python Environment

Install the required Python packages using `pip`:

```bash
pip install numpy astropy matplotlib reproject photutils astroquery
