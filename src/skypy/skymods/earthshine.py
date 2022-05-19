import pickle
import numpy as np
import pandas as pd
import pydap.client
import matplotlib.pyplot as plt
from os import path
from skypy.utils import package_directory, add_to_pickel_file, open_dap_url_constructor, get_ceres_dap_url_hourly_cadence, load_yaml, merge_two_dictionaries
from skypy.telescopes.hst import HubbleObservation, get_hst_ceres_fluxes_hourly_no_class
from skypy.telescopes.ceres import prep_data
from skypy.tles import pick_the_right_tle


class CeresSkyModel():
    """For a HST observation, determine what the upwards flux is under the telescope for a certain data product that your interested in.

    TLE's are included from 2000-01-01 to 2019-01-20. For other observations, please download them from celestrak and include them in tle_dictionary.yml
    This only works for hourly cadence data. If you want to use a different data set not supported for remote download by LARK, then download via https://ceres.larc.nasa.gov/data/
    and use the calculate_flux_under_hst notebook as a guide to dervice the flux under HST with a local ceres file.
    """

    def __init__(self, observation_start_date_time, observation_end_date_time, data_product="init_all_toa_sw_up"):
        """
            Inputs:
            ------
                observation_start_date_time (datetime.datetime): start time and date of hst observation  (from FITS header)
                observation_end_date_time (datetime.datetime): end time and date of hst observation (from FITS header)
                data_product (str): the ceres data product
        """
        self.obs_start = observation_start_date_time
        self.obs_end = observation_end_date_time
        self.obs_mid = (observation_start_date_time +
                        (observation_end_date_time - observation_start_date_time)/2)
        self.data_product = "init_all_toa_sw_up"
        # Define where the master TLE yaml file is
        self.tle_dictionary = load_yaml(package_directory('data', 'tle_dictionary.yml'))
        self.correct_tle = pick_the_right_tle(self.obs_start, self.tle_dictionary)

    def download_allsky_data(self, saveloc, data_product):
        """Get ceres data for the entire surface of the eath from ceres

            Inputs:
            ------
                saveloc (str): local save location for the data
                data_product (str): the name of the ceres data product you want (usually init_all_toa_sw_up)
        """
        # Get the file extension batch number for the CERES data base based on the HST obs date
        code_40 = open_dap_url_constructor(self.obs_start)

        if code_40 == -999:
            raise Exception("Sorry this does not exist")
        else:
            # Otherwise get the CERES URL in order to import the data remotely
            search_url = get_ceres_dap_url_hourly_cadence(code_40,
                                                          self.obs_start)
            hst_hour = self.obs_mid.hour - 1

            try:
                dataset = pydap.client.open_url(search_url)
                self.ceres_data = dataset[data_product][hst_hour, :, :]
                pickel_filename = "ceres_data_{}".format(self.obs_mid.strftime("%d%m%y_%H%M%S"))
                add_to_pickel_file(saveloc, pickel_filename,
                                   pickel_dictionary={'sw_upward_flux': self.ceres_data.data})

                return(self.ceres_data[0])

            except RuntimeError as error:
                print(error)

    def get_earthshine(self, telescope_altitude):
        """Use the data downloaded to get the Earthshine estimate below HST for the particular exposure

            Inputs:
            ------
                telescope_altitude (float): altitude of the telescope in km

            Outputs:
            -------
                out_dict (dict): a dictionary of the average ceres observed fluxes in W/m^2 and stdev
        """
        ceres_fluxes_dict = self.ceres_data.data[0, :, :]
        data_frame_organised = prep_data(ceres_fluxes_dict)
        fluxes_in_start, fluxes_in_end, fluxes_in_mid = get_hst_ceres_fluxes_hourly_no_class(self.obs_start,
                                                                                             self.obs_mid,
                                                                                             self.obs_end,
                                                                                             self.correct_tle,
                                                                                             data_frame_organised,
                                                                                             telescope_altitude)

        # Compute the average flux under hst
        flux_start_av = np.average(fluxes_in_start)
        flux_mid_av = np.average(fluxes_in_mid)
        flux_end_av = np.average(fluxes_in_end)

        # Compute the standard deviation in flux
        flux_start_stdv = np.std(fluxes_in_start)
        flux_mid_stdv = np.std(fluxes_in_mid)
        flux_end_stdv = np.std(fluxes_in_end)

        # Put it all in a dictionary
        out_dict = {'flux_start_av': flux_start_av,
                    'flux_mid_av': flux_mid_av,
                    'flux_end_av': flux_end_av,
                    'flux_start_stdv': flux_start_stdv,
                    'flux_mid_stdv': flux_mid_stdv,
                    'flux_eend_stdv': flux_end_stdv}
        return(out_dict)
