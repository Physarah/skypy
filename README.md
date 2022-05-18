# skypy

Skypy is a python package that is designed to make is easier to predict the intensity of the sky as observed from low Earth orbit.

There are a few prerequisites for the package and things you will need to install apart from what is in the requirements.txt. These are:

- basemap: https://matplotlib.org/basemap/users/installing.html
- pysynphot: https://pysynphot.readthedocs.io/en/latest/

You will also need to download the HST calibration files (CBDS) and set the path to these in the config/general.yaml which can be found at the above link. Zodi XGBoost models can be accessed in the repo without the need for the CBDS.
