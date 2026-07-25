# Astronomical Data Reduction Pipeline

A Python-based astronomical data reduction pipeline designed to calibrate raw science frames, align and stack images, plate-solve sky coordinates, and perform precision empirical Point Spread Function (ePSF) photometry.

---

## Image Calibration Formula

Raw science images are calibrated frame-by-frame using standard calibration equations:

$$\text{Calibrated Image} = \frac{\text{Raw Science} - \text{Master Dark}}{\left( \frac{\text{Raw Flat} - \text{Master Bias}}{\text{Normalization Factor}} \right)}$$

---

## Prerequisites & Installation

### 1. Windows Subsystem for Linux (WSL) & Astrometry.net
This pipeline relies on **Astrometry.net** running locally inside **Windows Subsystem for Linux (WSL)** to plate-solve coordinate headers. 

> 📌 **Installation Instructions:** > For complete step-by-step terminal commands on how to install WSL, set up Astrometry.net, configure optional D: drive storage, and download necessary index files, please follow the guide in [`astrometry_install.py`](./astrometry_install.py).

---

### 2. Python Environment
Install the required Python packages using `pip`:

```
pip install numpy astropy matplotlib reproject photutils astroquery
