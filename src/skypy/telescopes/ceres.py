import warnings
import re
import pandas as pd
import numpy as np
from datetime import datetime
from netCDF4 import Dataset
from pyhdf.SD import SD, SDC


def get_correct_ceres_file(ceres_folder_path, date_to_get):
    onlyfiles = [f for f in listdir(ceres_folder_path) if isfile(join(ceres_folder_path, f))]
    for i in onlyfiles:
        if not i == ".DS_Store":
            date_numbers = re.findall("[+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][+]?\d+)?", i)
            date_numbers_of_use = date_numbers[-2:]
            start_date = datetime.datetime.strptime(
                ''.join(c for c in date_numbers_of_use[0] if c.isdigit()), '%Y%m%d')
            end_date = datetime.datetime.strptime(
                ''.join(c for c in date_numbers_of_use[1] if c.isdigit()), '%Y%m%d')
            if start_date <= date_to_get <= end_date:
                return(i)


def get_days_since_ceres_launch(data_all):
    days_since_date = datetime.datetime.strptime('2000-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    dates_to_add = []
    for i in np.arange(0, len(data_all)):
        date_to_get = datetime.datetime.strptime(
            data_all['Data Start Time'].iloc[i], '%Y-%m-%d %H:%M:%S.%f')
        delta = date_to_get - days_since_date
        secs_per_day = 24*60*60    # hours * mins * secs
        date_number = delta.total_seconds()/secs_per_day
        dates_to_add.append(date_number)
    return(dates_to_add)


def calculate_path_for_all_data(data_all, ceres_folder_path):
    ceres_files = []
    for i in np.arange(0, len(data_all)):
        date_to_get = data_all['Data Start Time'].iloc[i]
        date_to_get = datetime.datetime.strptime(date_to_get, '%Y-%m-%d %H:%M:%S.%f')
        path = get_correct_ceres_file(ceres_folder_path, date_to_get)
        if path == None:
            pathout = -999
        else:
            pathout = path
        ceres_files.append(pathout)
    return(ceres_files)


def convert_ceres_file_date_to_datetime(ceres_file_name):
    """
    Gets the CERES dates from the file name and converts them into datetime objects for the start and end of a ceres observations.
    This is basically to check, before the import the file into a ceres class that the data in here corresponds to the
    observing time that Hubble has used.

    Inputs:
    ------
        ceres_file_name: (str) a string corresponding to the single file name or full path to a ceres file

    Outputs:
    -------
        start_of_obs_datetime: (datetime.datetime) a datetime object of the start of a ceres observation
        end_of_obs_datetime: (datetime.datetime) a datetime object of the end of a ceres observation
    """
    read_file_type = ceres_file_name[0:17]
    if read_file_type != 'CERES_SYN1deg-Day':
        warnings.warn(
            "Wait! You're not using the correct CERES data product. I'm currently reading {}. You want CERES_SYN1deg-Day".format(read_file_type))

    else:
        date_numbers = re.findall(
            "[+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][+]?\d+)?", ceres_file_name)
        date_numbers_of_use = date_numbers[-2:]
        start_of_obs = str(int(float(date_numbers_of_use[0])))
        end_of_obs = str(int(float(date_numbers_of_use[1])))
        format = '%Y%m%d'
        start_of_obs_datetime = datetime.strptime(start_of_obs, format)
        end_of_obs_datetime = datetime.strptime(end_of_obs, format)
        return(start_of_obs_datetime, end_of_obs_datetime)


def get_ceres_data_products_names(ceres_file_path):
    """
    Quick look function for a ceres data set. It returns all the names of the veriables in the netcdf4 file
    in a list format for quick reading.

    Inputs:
    ------
        ceres_file_name: (str) a string corresponding to the full path to a ceres file

    Outputs:
    -------
        list_of_var: (list) a list of the variables in the ceres netcdf4 file as strings
    """
    list_of_var = []
    nc = Dataset(ceres_file_path)
    for key in nc.variables.keys():
        list_of_var = np.append(list_of_var, key)
    return(list_of_var)


def get_ceres_weather_data_for_given_date(ceres_data_file, field_name, date_integer_python):
    """
    Get the longitude and latitude and flux from a ceres file for a speficied date in a month

    Inputs:
    ------
        ceres_data_file: (netCDF4._netCDF4.Dataset) the ceres loaded data frame i.e. Dataset(file_path)
        field_name: (str) the name of the data product you want to get out e.g. 'toa_sw_all_daily'
        date_integer_python: (int) this is the integer of the date you want to get out. Generally, 1 - the calendar day
            to accomodate for netcdf4 indexing from 0

    Outputs:
    -------
        longitude: () all the ceres longitudes
        latitude: () all the ceres latitudes
        field_data: () all ceres field values (like fluxes) for chosen field_name
    """
    var = ceres_data_file.variables[field_name]
    data = var[date_integer_python, :, :].astype(np.float64)
    latitude = ceres_data_file.variables['lat'][:]
    longitude = ceres_data_file.variables['lon'][:]
    invalid = np.logical_or(data < float(var.valid_min),
                            data > float(var.valid_max))
    data[invalid] = np.nan
    field_data = np.ma.masked_array(data, mask=np.isnan(data))
    return(longitude, latitude, field_data)


def get_ceres_data_object(ceres_file_path, day_index, variable):
    """
    Gets the ceres dataframe object for all longitudes and latitudes, for a specified date.

    Inputs:
    ------
        ceres_file_path: (str) the path to the netcdf4 file from ceres
        day_index: (int) the index of the day in the month that is required. It is 1 - the date
        variable: (str) the dataproduct of interest, for example 'toa_sw_all_daily'

    Output:
    ------
        data_object: (numpy.ma.core.MaskedArray) the object containing all the info for that variable
    """
    dataframe = Dataset(ceres_file_path)
    specified_data_frame = dataframe.variables[variable]
    data_object = specified_data_frame[day_index, :, :].astype(np.float64)
    return(data_object)


def prep_data(masked_ceres_data):
    list_thing_lats = (((np.arange(0.5, 180, 1)) - 90))
    list_thing_longs = (((np.arange(0.5, 360, 1)) - 180))
    lat_list = np.arange(0, 179, 1)
    lon_list = np.arange(0, 359, 1)
    organised_data = []
    for i in lat_list:
        for j in lon_list:
            data_point = masked_ceres_data[i, j]
            long = list_thing_longs[j]
            lat = list_thing_lats[i]
            organised_data.append([lat, long, data_point])
    df = pd.DataFrame(organised_data, columns=['Latitude', 'Longitude', 'Variable'])
    return(df)


def read_and_parse_ceres_file(ceres_data_file, thing_we_want):
    """
    thing_we_want = "init_all_toa_sw_up"
    ceres_data_file = "/Users/physarah/Downloads/CER_SYN1deg-Day_Terra-Aqua-MODIS_Edition4A_401405.20020718"
    """
    hdf_file = SD(ceres_data_file, SDC.READ)
    data3D = hdf_file.select(thing_we_want)
    data = data3D[:, :].astype(np.float64)
    attrs = data3D.attributes(full=1)
    fva = attrs["_FillValue"]
    fillvalue = fva[0]
    data[data == fillvalue] = np.nan
    datam = np.ma.masked_array(data, mask=np.isnan(data))
    return(datam)


def read_and_parse_ceres_file_for_clouds(ceres_data_file, thing_we_want):
    """
    thing_we_want = "init_all_toa_sw_up"
    ceres_data_file = "/Users/physarah/Downloads/CER_SYN1deg-Day_Terra-Aqua-MODIS_Edition4A_401405.20020718"
    """
    hdf_file = SD(ceres_data_file, SDC.READ)
    data3D = hdf_file.select(thing_we_want)
    data_high_cloud = data3D[5, :, :].astype(np.float64)
    attrs = data3D.attributes(full=1)
    fva = attrs["_FillValue"]
    fillvalue = fva[0]
    data_high_cloud[data_high_cloud == fillvalue] = np.nan
    datam = np.ma.masked_array(data_high_cloud, mask=np.isnan(data_high_cloud))
    return(datam)


def read_and_parse_ceres_file_no_nan(ceres_data_file, thing_we_want):
    hdf_file = SD(ceres_data_file, SDC.READ)
    data3D = hdf_file.select(thing_we_want)
    data = data3D[:, :].astype(np.float64)
    attrs = data3D.attributes(full=1)
    fva = attrs["_FillValue"]
    fillvalue = fva[0]
    return(data)


def get_ceres_data_all(ceres_data_file, thing_we_want):
    hdf_file = SD(ceres_data_file, SDC.READ)
    data3D = hdf_file.select(thing_we_want)
    data = data3D[:, :].astype(np.float64)
    attrs = data3D.attributes(full=1)
    fva = attrs["_FillValue"]
    fillvalue = fva[0]
    data[data == fillvalue] = np.nan
    datam = np.ma.masked_array(data, mask=np.isnan(data))
    longitude = hdf_file.select('longitude')
    latitude = hdf_file.select('latitude')
    latitude_out = latitude[:].astype(np.float64)
    longitude_out = longitude[:].astype(np.float64)
    return(longitude_out, latitude_out, datam)
