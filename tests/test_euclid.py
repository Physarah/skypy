import pytest
import datetime
import astropy.units as u
from astropy.coordinates import SkyCoord
from skypy.skymods.euclid import EuclidSkyModel


def test_euclid():
    model = EuclidSkyModel(coordinates=SkyCoord('0h39m15.9s', '0d53m17.016s', frame='icrs'),
                           wavelength=200*u.micron,
                           date=datetime.datetime(2019, 4, 13),
                           observing_location='L2',
                           code_version='Wright',
                           median=True)

    if model.zodiacal_light == 12.55276 * u.MJy * u.sr**-1:
        assert True
    else:
        assert False
