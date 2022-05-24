# Skypy

Skypy is a python package that is designed to make is easier to predict the intensity of the sky as observed from low Earth orbit.

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

To run the jupyter notebook examples, you can install jupyter via conda
```
conda install jupyter
```

To start the jupyter-notebook, type `jupyter-notebook` in your terminal, in the skypy environment where the above dependancies are installed

## Config Files 

There is one set of config files not included in this repo that you will need to download if you want to produce SED's and sky backgrounds for HST, and that is the HST CDBS. You can find them at https://archive.stsci.edu/hlsps/reference-atlases/cdbs/

If these ever disapear, you can contact the author of this software package for a coby of the originals. It has not been included on this repo because they total about 4GB. 

Once you have downloaded the file, point the package to where it is stored by editing the `general.yaml` file, using the PYSYN_CDBS keyword. E.g.:
```
PYSYN_CDBS: "/Volumes/CODE/Hubble/grp/hst/cdbs"
```
In `general.yaml` add the path to the example data provided in `.../src/skypy/data/hst_skybackgrounds.csv`

```
data_path: "/Users/physarah/Development/skypy/src/skypy/data/hst_skybackgrounds.csv"
```
