from skypy.utils import load_config, package_directory
from skypy.skymods.xgboost import XGBoostSkyModel
import astropy.units as u
import pandas as pd
import numpy as np
import pytest


def test_model():
    sky = XGBoostSkyModel()
    sky.prepare_training_data('F606W')
    sky.train()
    sky_value = sky.predict('field', 30, 30, 30, 30, 30, 30, 30, 200,
                            average=True, plot=False,
                            low_lim=0.3, high_lim=0.8)
    if sky_value.value == pytest.approx(0.47607046, 0.01):
        assert True
    else:
        assert False
