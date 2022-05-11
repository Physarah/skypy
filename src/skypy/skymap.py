from skypy.utils import load_config, package_directory

from loguru import logger as log
from astropy.coordinates import SkyCoord
from sklearn.model_selection import train_test_split
from mpl_toolkits.basemap import Basemap

import matplotlib as mpl
import xgboost as xgb
import astropy.units as u
import pandas as pd
import numpy as np
import warnings
import healpy as hp


class SkyModel():
    """A class to train and test sky models using HST data

    """

    def __init__(self):
        """Simple init to load config and shared attributes

        Inputs:
        ------
            data_path (str): path to csv file
            g_clip (float): +- in degrees to clip on the galatic plane. If False, there is no clip
            sigma (float): How many outliars to clips. If False, no clip

        Returns:
        -------
            out_dict (dict): dictionary of fluxes etc for each filter after processing
        """
        # load config
        config_path = package_directory(local_path='config', filename='mlconfig.yaml')
        self.config = load_config(config_path)

        # load data from wonderful people at STScI
        with_zodi_model = pd.read_csv(self.config['data_clean']['data_path'])

        # get rid of known bright sources, unknowns, and issues
        target_filter_list = self.config['bad_fields']

        for i in target_filter_list:
            with_zodi_model = with_zodi_model[~with_zodi_model['Target Name'].str.contains(i)]

        # get galatic coordinates
        c_gal1 = SkyCoord(with_zodi_model['GLON_REF'],
                          with_zodi_model['GLAT_REF'], unit='deg', frame='galactic')

        # wrap coordinates to deal with 0 - 360
        g_longs = c_gal1.l.wrap_at(180 * u.deg).deg
        g_lats = c_gal1.b.deg
        with_zodi_model['GLAT_REF'] = list(g_lats)
        with_zodi_model['GLON_REF'] = list(g_longs)

        # if you want to clip the galatic plane (we generally don't for a more complete model)
        if self.config['data_clean']['clip_galatic']:
            with_zodi_model = with_zodi_model[(with_zodi_model['GLAT_REF'] > g_clip) | (
                with_zodi_model['GLAT_REF'] < -g_clip)]
        else:
            log.info("Not clipping data on the galatic plane")
            pass

        # breakup into filters
        out_dict = {}

        try:
            out_dict['F606W'] = with_zodi_model[with_zodi_model['FILTER1'] == 'F606W']
        except:
            log.info("No F606W data here")
        try:
            out_dict['F814W'] = with_zodi_model[with_zodi_model['FILTER2'] == 'F814W']
        except:
            log.info("No F814W data here")
        try:
            out_dict['F435W'] = with_zodi_model[with_zodi_model['FILTER2'] == 'F435W']
        except:
            log.info("No F435W data here")
        try:
            out_dict['F850LP'] = with_zodi_model[with_zodi_model['FILTER1'] == 'F850LP']
        except:
            log.info("No F606W data here")

        # for diagnosis of issues, you can sigma clip if you like
        if self.config['data_clean']['sigma_clip']:
            for key, value in out_dict.items():

                out_dict[key] = value[(value['sky_sb'] == list(sigma_clip(value['sky_sb'],
                                                                          sigma=sigma,
                                                                          maxiters=5)))]
        else:
            log.info("Not sigma clipping data")

        self.data_dictionary = out_dict

    def prepare_training_data(self,
                              hst_filter,
                              test_set_size=0.3,
                              random_state=13):
        """Split into predictors, and labeled data,

        Inputs:
        ------
            training_data (pandas.core.frame.DataFrame): pandas dataframe for a specific filter
            test_set_size (float): fraction of testing vs training data
            random_state (float): seed for test/train split

        Returns:
        -------
            out_dict (dict): dictionary of fluxes etc for each filter after processing
        """

        training_data = self.data_dictionary[hst_filter]
        predictors = training_data[self.config['predictors']]
        self.hst_filter = hst_filter

        # set the labaled data to predict
        sky_background = training_data[self.config['label']]

        # do a test train slipt
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            predictors, sky_background, test_size=test_set_size, random_state=random_state)

    def train(self, x_training_data=False, y_training_data=False):
        """Train an xgboost model

        Inputs:
        ------
            x_training_data (pandas.core.frame.DataFrame): the training data - predictors
            y_training_data (pandas.core.frame.DataFrame): the labled data - sky backgrounds
        """

        # init an xgboost regression model
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror',
                                  n_estimators=self.config['model_params']['n_estimators'],
                                  learning_rate=self.config['model_params']['learning_rate'],
                                  early_stopping_rounds=self.config['model_params']['early_stopping_rounds'],
                                  verbose=0)

        # fit the model to the training data
        if x_training_data:
            xg_reg.fit(x_training_data, y_training_data, verbose=0)
        else:
            log.info("Using pre-calculated training data")
            xg_reg.fit(self._X_train, self._y_train, verbose=0)

        self.model = xg_reg

    def predict(self,
                type,
                exp_time,
                moon_angle,
                limb_angle,
                sun_alt,
                sun_angle,
                gal_lat,
                gal_long,
                n_samples=None,
                average=False,
                plot=False,
                low_lim=False,
                high_lim=False):
        """Use the learner to predict the sky background values for some test data

        Inputs:
        ------
            type (str): can be "allsky" or "field". This will specifiy if the whole sky is sampled, or just a specific field
            exp_time (float): exposure time (s)
            g_lat (float): galatic latitude (deg). If type = allsky, set to False
            g_long (float): galatic longiude (deg). If type = allsky, set to False
            moon_angle (float): angle between moon and field (deg)
            limb_angle (float): angle between Earth's limb and field (deg)
            sun_alt (float): sun altitude (deg)
            sun_angle (float): sun angle (deg)
            n_samples (float): how many subsamples to create on an allsky grid (if allsky is selected)if this is too low the plot will look odd due to the random sampling
            average (bool): if average = True, 10 iteration will be calculated with randomly sampled points and the result will be averaged
            plot (bool): if true will plot the output
            low_lim (float): cbar lower lim for plot
            high_lim (float): cbar upper lim for plot

        Outputs:
        -------
            single_field_prediction (astropy.units): the predicted single field sky brightness
            multi_field_prediciton (astropy.units): the predicted total sky brightness # check

        """
        # single field predicition
        if type == "field":
            # put into xgboost format
            X_test = pd.DataFrame({'EXPTIME': [exp_time],
                                   'GLAT_REF': [gal_lat],
                                   'GLON_REF': [gal_long],
                                   'LOS_MOON': [moon_angle],
                                   'Earth Limb Angle Range (deg):LOS_LIMB': [limb_angle],
                                   'SUN_ALT': [sun_alt],
                                   'SUNANGLE': [sun_angle]})
            # predict
            single_field_prediction = self.model.predict(X_test)

            return(single_field_prediction[0]*u.MJy * u.sr**-1)

        # allsky predicition
        elif type == "allsky":

            if not average:

                log.info("Just doing one run, no average")

                # create a randomly sampled grid on the sky of n_samples size
                lats_to_use = np.random.uniform(-90, 90, size=(n_samples))
                longs_to_use = np.random.uniform(-180, 180, size=(n_samples))

                # for each sample make a prediction
                hst_flux_list = []
                for i in np.arange(0, n_samples):
                    g_lat = lats_to_use[i]
                    g_long = longs_to_use[i]
                    hst_flux_prediction = self.model.predict(pd.DataFrame({'EXPTIME': [exp_time],
                                                                           'GLAT_REF': [g_lat],
                                                                           'GLON_REF': [g_long],
                                                                           'LOS_MOON': [moon_angle],
                                                                           'Earth Limb Angle Range (deg):LOS_LIMB': [limb_angle],
                                                                           'SUN_ALT': [sun_alt],
                                                                           'SUNANGLE': [sun_angle]}))
                    hst_flux_list.append(hst_flux_prediction[0])

                self.multi_field_prediciton = hst_flux_list
                return(self.multi_field_prediciton)

            else:

                log.info("Doing 10 runs, will take the average")
                # uniform grid to resample the random points (random points ensure lack of bias)
                xi = np.linspace(-180, 180, n_samples)
                yi = np.linspace(-90, 90, n_samples)
                flux_arrays = []
                for i in np.arange(0, 10):

                    # create a randomly sampled grid on the sky of n_samples size
                    self._lats_to_use = np.random.uniform(-90, 90, size=(n_samples))
                    self._longs_to_use = np.random.uniform(-180, 180, size=(n_samples))

                    # for each sample made a prediction
                    hst_flux_list = []
                    for i in np.arange(0, n_samples):
                        # we don't loop over j and i, because they are random points, not a meshgrid (yet)
                        g_lat = self._lats_to_use[i]
                        g_long = self._longs_to_use[i]
                        hst_flux_prediction = self.model.predict(pd.DataFrame({'EXPTIME': [exp_time],
                                                                               'GLAT_REF': [g_lat],
                                                                               'GLON_REF': [g_long],
                                                                               'LOS_MOON': [moon_angle],
                                                                               'Earth Limb Angle Range (deg):LOS_LIMB': [limb_angle],
                                                                               'SUN_ALT': [sun_alt],
                                                                               'SUNANGLE': [sun_angle]}))
                        # remove list stuff
                        hst_flux_list.append(hst_flux_prediction[0])

                    # just ensure everything is a float for numpy
                    new_hst_fluxes = []
                    for i in hst_flux_list:
                        new_hst_fluxes.append(np.float(i))

                    # linearly interpolate the data (x, y) on a grid defined by (xi, yi).
                    triang1 = mpl.tri.Triangulation(self._longs_to_use, self._lats_to_use)
                    interpolator1 = mpl.tri.LinearTriInterpolator(triang1, new_hst_fluxes)
                    Xi1, Yi1 = np.meshgrid(xi, yi)
                    zi1 = interpolator1(Xi1, Yi1)
                    flux_arrays.append(zi1)

                if plot:
                    log.info("Plotting output")
                    norm = mpl.colors.Normalize(vmin=low_lim, vmax=high_lim)
                    m = Basemap(projection='moll', lon_0=360, resolution='c')
                    # get the average of the looped arrays
                    colormesh = m.contourf(Xi1, Yi1, np.average(
                        flux_arrays, axis=0), 30, cmap='magma', latlon=True, norm=norm)
                    cbar = mpl.pyplot.colorbar(colormesh, shrink=0.8, label='Sky SB MJy/sr',
                                               norm=norm, orientation='horizontal')
                    cbar.set_clim(low_lim, high_lim)

                    mpl.pyplot.title(self.hst_filter)
                    mpl.pyplot.axis("off")
                    mpl.pyplot.savefig(self.config['saveloc'])

                else:
                    log.info("Not plotting output")
                    return(np.average(flux_arrays, axis=0))

        else:
            warnings.error("Incorrect type input, only accepts 'allsky' or 'field'")

    def save_healpix():
        pass
