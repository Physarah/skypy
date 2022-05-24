# Skypy

Skypy is a python package that is designed to make is easier to predict the intensity of the sky as observed from low Earth orbit.

There are a few prerequisites for the package and things you will need to install apart from what is in the requirements.txt. These are:

You will also need to download the HST calibration files (CBDS) and set the path to these in the config/general.yaml which can be found at the above link. Zodi XGBoost models can be accessed in the repo without the need for the CBDS.

## Installation 

This software has been tested in python 3.8 in macOS 10.14.6 in an anaconda environment (4.10.1)

In a fresh environment install xgboost using conda to avoid issues with cmake 

```
conda install xgboost=1.5.0
```

To install Skypy, first clone it 
```
git clone https://github.com/Physarah/skypy.git
```

The navigate to `.../src/skypy` and run 
```
python setup.py install 
python setup.py develop
```
You can now install the dependencies 
```
pip install -r requirements.txt 
```

If you want to plot some of the results, you'll need to also install basemap: https://matplotlib.org/basemap/users/installing.html The easiest way to do this is via conda:
```
conda install basemap 
```
Note: this package is depreciated, so this will install the last version which is up to date for Skypy.

Pysynphot https://pysynphot.readthedocs.io/en/latest/ has also depreciated, but the correct version (2.0.0) can be installed via the requirements.txt with pip. 


