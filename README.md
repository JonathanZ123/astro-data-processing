# Astronomical Data Reduction Pipeline

Python-based astronomical data reduction pipeline designed to calibrate, align, stack, and calculate instrumental/apparent flux using ePSF photometry and local Astrometry.net plate-solving.

## Image Calibration Formula

The raw science images are calibrated frame-by-frame using the standard reduction equation:

$$\text{Calibrated Image} = \frac{\text{Raw Science} - \text{Master Dark}}{\left( \frac{\text{Raw Flat} - \text{Master Bias}}{\text{Normalization Factor}} \right)}$$

---

## Prerequisites & Installation

Before running `astro_data_reduction.py`, ensure you have Python installed along with WSL (for Astrometry.net plate-solving). 

Install all required Python packages with:

```bash
pip install numpy astropy matplotlib reproject photutils astroquery
