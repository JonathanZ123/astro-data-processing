Python based astronomical data reduction pipeline designed to calibrate, align, stack, and calculate flux using psf and astrometry.net.
<br>
$$Calibrated\ Image = \frac{Raw\ Science - Master\ Dark}{\left(\frac{Raw\ Flat - Master\ Bias}{Normalization\ Factor}\right)}$$
<br>
## Prerequisites & Installation

Before running `astro_data_reduction.py`, ensure you have Python installed along with WSL (for Astrometry.net plate-solving). 

Install all required Python packages with:

```bash
pip install numpy astropy matplotlib reproject photutils astroquery
