import astropy.units as u
import numpy as np
import warnings
import pickle
import calendar
import yaml
import os


def load_config(config_file):
    """
    Load config file

    Params:
        config_file: str
            yaml file name in config
    """
    config_path = package_directory(local_path='config', filename=f'{config_file}.yaml')
    with open(config_path, 'r') as stream:
        try:
            outdict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return(outdict)


def load_yaml(path):
    """
    Load yaml file

    Params:
        path: str
            yaml file path
    """
    with open(path, 'r') as stream:
        try:
            outdict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        return(outdict)


def package_directory(local_path, filename):
    """Get the directory where package is installed for data

    Inputs:
    ------
        local_path (str): path to file inside package
        filename (str): name of file

    Returns:
    -------
        full_path (str): full path on users system to any file in package
    """
    file_path = os.path.dirname(os.path.realpath(__file__))
    full_path = os.path.join(file_path, local_path, filename)
    return(full_path)


def setup_cdbs():
    if not('PYSYN_CDBS' in os.environ.keys()):  # nopep8
        os.environ['PYSYN_CDBS'] = get_cbds_filepath()  # nopep8
    import pysynphot as S
    return(S)


def get_cbds_filepath():
    config = load_config('general')
    cbds_path = config['PYSYN_CDBS']
    return(cbds_path)


def wfc3_abmag_zeropoint(photflam):
    """
    http://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
    """
    wfc3_abmag_zpt = -2.5 * np.log10(photlam)
    return(wfc3_abmag_zpt)


def acs_abmag_zeropoint(photlam, photplam):
    """
    http://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints
    """
    acs_abmag_zpt = -2.5 * np.log10(photlam) - 5 * np.log10(photplam) - 2.408
    return(acs_abmag_zpt)


def rate_to_flam(count_rate, photflam):
    """
    Calculate the flux density and stick astropy units on it
    """
    sky_flux_density = count_rate * photflam
    return(sky_flux_density)


def flux_density_to_photon_flux(flux_density, pivot_wavelength):
    """
    Convert the flux density to a photon flux with the pivot wavelength
    """
    photon_flux = flux_density.to(u.photon * u.s**-1 * u.cm**-2 * u.Angstrom**-1,
                                  equivalencies=u.equivalencies.spectral_density(pivot_wavelength))
    return(photon_flux)

    import yaml


def select_file_in_dir(hst_obs_object, directory_with_ceres_files="/Users/physarah/Downloads"):
    onlyfiles = [f for f in os.listdir(directory_with_ceres_files)
                 if os.path.isfile(os.path.join(directory_with_ceres_files, f))]
    date_time_string = hst_obs_object.start_times.strftime("%Y%m%d")
    matching = [s for s in onlyfiles if date_time_string in s]
    return(matching)


def handel_met_files_and_doubles(ceres_possible_matches):
    no_met_list = [x for x in ceres_possible_matches if ".met" not in x]
    no_txt_list = [x for x in no_met_list if ".txt" not in x]
    if len(no_txt_list) > 1:
        correct_file = no_txt_list[0]
        return(correct_file)
    else:
        return(no_txt_list)


def open_dap_url_constructor(hst_start_time):

    hst_obs_year = hst_start_time.year
    hst_obs_month = hst_start_time.month
    comp_arry = (hst_obs_year, hst_obs_month)

    arr1_comp_1 = (2002, 7)
    arr1_comp_2 = (2016, 8)

    arr2_comp_1 = (2016, 9)
    arr2_comp_2 = (2017, 2)

    arr3_comp_1 = (2017, 3)
    arr3_comp_2 = (2018, 3)

    arr4_comp_1 = (2018, 4)
    arr4_comp_2 = (2020, 3)

    if arr1_comp_1 <= comp_arry <= arr1_comp_2:
        code_40 = '401405'
    else:
        if arr2_comp_1 <= comp_arry <= arr2_comp_2:
            code_40 = '402405'
        else:
            if arr3_comp_1 <= comp_arry <= arr3_comp_2:
                code_40 = '403405'
            else:
                if arr4_comp_1 <= comp_arry <= arr4_comp_2:
                    code_40 = '407406'
                else:
                    code_40 = -999
                    warnings.warn("This date is outside of the CERES data set")
    return(code_40)


def get_ceres_dap_url_daily_cadence(code_40, hst_obs_date):
    larc_entry_point = "https://opendap.larc.nasa.gov/opendap/CERES/SYN1deg-Day/Terra-Aqua-MODIS_Edition4A/"
    hst_day = '{:02d}'.format(hst_obs_date.day)
    hst_month = '{:02d}'.format(hst_obs_date.month)
    hst_year = str(hst_obs_date.year)
    date_directory = hst_year + "/" + hst_month
    data_product = "/CER_SYN1deg-Day_Terra-Aqua-MODIS_Edition4A_"
    code_40_dir = code_40 + "." + hst_year + hst_month + hst_day + ".hdf"
    return(larc_entry_point + date_directory + data_product + code_40_dir)


def get_ceres_dap_url_hourly_cadence(code_40, hst_obs_date):
    larc_entry_point = "https://opendap.larc.nasa.gov/opendap/CERES/SYN1deg-1Hour/Terra-Aqua-MODIS_Edition4A/"
    hst_day = '{:02d}'.format(hst_obs_date.day)
    hst_month = '{:02d}'.format(hst_obs_date.month)
    hst_year = str(hst_obs_date.year)
    date_directory = hst_year + "/" + hst_month
    data_product = "/CER_SYN1deg-1Hour_Terra-Aqua-MODIS_Edition4A_"
    code_40_dir = code_40 + "." + hst_year + hst_month + hst_day + ".hdf"
    return(larc_entry_point + date_directory + data_product + code_40_dir)


def dump_dictionary_to_yaml(file_name, dictionary):
    """
    Dump current configuration (dictionary) to a file in the configuration directory
    of the current experiment.

    Inputs:
    ------
    file_name : str
        Name of the .yml file to dump to.

    """
    config = load_config('data')
    dump_path = os.path.join(config['tle_files']['tle_storage_directory'], file_name + '.yml')
    with open(dump_path, 'w') as f:
        yaml.dump(dictionary, f)


def get_tle_archive_path():
    """
    Gets the tle path input by the user from the config file
    """
    config = load_config('data')
    tle_path = config['tle_files']['tle_directory']
    return(tle_path)


def get_pickel_path():
    """
    Gets the pickle path input by the user from the config file.
    This might be where you want to save the pickled data, if you are scripting things 
    """
    config = load_config('data')
    pick_path = config['pickel_directory']
    return(pick_path)


def get_tle_storage_path():
    """
    Gets the tle storage path where the user can save current hst tle's
    """
    config = load_config('data')
    tle_storage_path = config['tle_files']['tle_storage_directory']
    return(tle_storage_path)


def get_pysynphot_external_data_path():
    """
    Because pysynphot requires an exteral data directory to work, and it's often difficult to use,
    I've set up a way of bypassing this diffculty by just updating a yaml to easily point to
    the extrernal data directory. This function just retreives it for imports.

    Returns:
    ------
        pysynphot_external_data_directory: (str) the location of the external data
    """
    config = load_config('general')
    pysynphot_external_data_directory = config['PYSYN_CDBS']
    return(pysynphot_external_data_directory)


def add_to_pickel_file(pickel_path, pickel_filename, pickel_dictionary):
    with open(pickel_path + "/" + pickel_filename + ".pickle", 'wb') as wfp:
        pickle.dump(pickel_dictionary, wfp)


def merge_two_dictionaries(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def totimestamp(d):
    return(calendar.timegm(d.timetuple()))


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
