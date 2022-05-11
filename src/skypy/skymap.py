from skypy.utils import load_config, package_directory
from loguru import logger as log

import pandas as pd
import numpy as np


class SkyModel():
    """A class to train and test sky models using HST data

    """
    def __init__():
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
        self.config = load_config(package_directory() + "/config/mlconfig.yaml")

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
            for key, value in out_dict.items()

            out_dict[key] = value[(value['sky_sb'] == list(sigma_clip(value['sky_sb'],
                                                                      sigma=sigma,
                                                                      maxiters=5)))]
            else:
                log.info("Sigma clipping data")

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

        # set the labaled data to predict
        sky_background = training_data[self.config['label']]

        # do a test train slipt
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            predictors, sky_background, test_size=test_set_size, random_state=random_state)

    def model_trainer(self, x_training_data, y_training_data):
        """Train an xgboost model

        Inputs:
        ------
            x_training_data (pandas.core.frame.DataFrame): the training data - predictors
            y_training_data (pandas.core.frame.DataFrame): the labled data - sky backgrounds
        """
        # init an xgboost regression model
        xg_reg = xgb.XGBRegressor(objective='reg:squarederror',
                                  n_estimators=10000,
                                  learning_rate=0.01,
                                  early_stopping_rounds=20)

        # fit the model to the training data
        xg_reg.fit(x_training_data, y_training_data)

        self.model = xg_reg

    def get_prediction(self,
                       learner,
                       exp_time,
                       g_lat,
                       g_long,
                       moon,
                       limb,
                       sun_alt,
                       sun_angle):
        """Use the learner to predict the sky background values for some test data,

        """

        X_test = pd.DataFrame({'EXPTIME': exp_time,
                               'GLAT_REF': g_lat,
                               'GLON_REF': g_long,
                               'LOS_MOON': moon,
                               'Earth Limb Angle Range (deg):LOS_LIMB': limb,
                               'SUN_ALT': sun_alt,
                               'SUNANGLE': sun_angle})

        pred = learner.predict(X_test)
        return(pred, X_test)

    def get_surface_brightnes_map(self, number=40, filters='F814W', wheel="FILTER2", exp_time=1000, moon=60, limb=80, sun_alt=-10, sun_angle=50):

        x_training_data, x_testing_data, y_training_data, y_testing_data = prepare_training_data(training_data=with_zodi_model,
                                                                                                 test_set_size=0.1,
                                                                                                 random_state=123)

        learner = model_trainer(x_training_data, y_training_data)

        lats_to_use = np.random.uniform(-90, 90, size=(number))
        longs_to_use = np.random.uniform(-180, 180, size=(number))
        flux_list = []
        for i in np.arange(0, number):
            exp_time = exp_time
            g_lat = lats_to_use[i]
            g_long = longs_to_use[i]
            moon = moon
            limb = limb
            sun_alt = sun_alt
            sun_angle = sun_angle

            hst_flux, params = get_prediction(learner=learner,
                                              exp_time=[exp_time],
                                              g_lat=[g_lat],
                                              g_long=[g_long],
                                              moon=[moon],
                                              limb=[limb],
                                              sun_alt=[sun_alt],
                                              sun_angle=[sun_angle])
            flux_list.append(hst_flux)

        fluxes = []
        for i in flux_list:
            fluxes.append(i[0])

        return(fluxes, lats_to_use, longs_to_use, x_training_data, x_testing_data)

    def interpolate(self, wheel, filters, exptime, moon, limb, sun_alt, sun_angle):
        flux_arrays_filter = []
        for i in np.arange(0, 10):

            flux_filter, lats_filter, longs_filter, training_filter, testing_filter = get_surface_brightnes_map(number=1000,
                                                                                                                filters=filters,
                                                                                                                wheel=wheel,
                                                                                                                exp_time=exptime,
                                                                                                                moon=moon,
                                                                                                                limb=limb,
                                                                                                                sun_alt=sun_alt,
                                                                                                                sun_angle=sun_angle)
            xi = np.linspace(-180, 180, 100)
            yi = np.linspace(-90, 90, 100)
            new_filter_fluxes = []
            for i in flux_filter:
                new_filter_fluxes.append(np.float(i))
            # # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
            triang1 = tri.Triangulation(longs_filter, lats_filter)
            interpolator1 = tri.LinearTriInterpolator(triang1, new_filter_fluxes)
            Xi1_filter, Yi1_filter = np.meshgrid(xi, yi)
            zi1_filter = interpolator1(Xi1_filter, Yi1_filter)

        flux_arrays_filter.append(zi1_filter)

    def plot_projection(self, xi, yi):

    new_814_fluxes = []
    for i in flux_F814W:
        new_814_fluxes.append(np.float(i))
        Xi1_f814w, Yi1_f814w = np.meshgrid(xi, yi)
        zi1_f814w = interpolator1(Xi1_f814w, Yi1_f814w)

        norm = mpl.colors.Normalize(vmin=0.1, vmax=0.4)
        m = Basemap(projection='moll', lon_0=360, resolution='c')
        colormesh = m.contourf(Xi1_f814w, Yi1_f814w, np.average(
            flux_arrays_f814w, axis=0), 30, cmap='magma', latlon=True, norm=norm)
        cbar = plt.colorbar(colormesh, shrink=0.8, label='Sky SB MJy/sr',
                            norm=norm, orientation='horizontal')
        cbar.set_clim(0.1, 0.4)

        plt.title('F814W')
        plt.axis("off")
