import astropy.units as u
import numpy as np
import pandas as pd
from math import radians, degrees, sin, cos, asin, acos, sqrt, atan2


def calculate_distance_to_horizon(altitude, planet_radius=6371 * u.km):
    """
    Simple function to calculate the distance to the horizon from an altitude above the earths surface

    Inputs
    ------
        altitude: (astropy.units.quantity.Quantity) generally the altitude of the spacecraft above the earths surface in km
        planet_radius: (astropy.units.quantity.Quantity) the radius of the planet the spacecraft is orbiting in km

    Returns
    ------
        horizon_distance: (astropy.units.quantity.Quantity) the distance to the horizon on the surface of the Earth. i.e (not the straight line distance)
    """
    horizon_fraction = planet_radius/(planet_radius + altitude.to(u.km))
    horizon_distance = planet_radius * np.arccos(horizon_fraction)
    return(horizon_distance * u.rad**-1)


def calcuate_great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Approximation:
    https://medium.com/@petehouston/calculate-distance-of-two-locations-on-earth-using-python-1501b1944d97
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    distance = 6371 * (acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1 - lon2)))
    return(distance * u.km)


def calculate_haversine_great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Most accurate method - 2 points on a sphere:
    https://medium.com/@petehouston/calculate-distance-of-two-locations-on-earth-using-python-1501b1944d97
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * 6371 * asin(sqrt(a))
    return(distance * u.km)


def vector_calculate_great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Get the great circle distance between alot of points in a numpy array
    https://gist.github.com/rochacbruno/2883505

    Inputs:
    ------
        lon1: (numpy.ndarray) longitude of point 1
        lat1: (numpy.ndarray) latitude of point 1
        lon2: (numpy.ndarray) longitude of point 1
        lat2: (numpy.ndarray) latitude of point 2

    Outputs:
    -------
        distance: (numpy.ndarray) distance between two points in kilometers
    """
    lat1 = lat1*np.pi/180.0
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    d = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin((lon2 - lon1)/2)**2
    distance = 2 * 6373.0 * np.arcsin(np.sqrt(d))
    return(distance)


def calculate_baring_between_two_points(lon1, lat1, lon2, lat2):
    """
    Calculate the baring or angle between two points on the globe, not vectorised
    """
    baring = atan2(cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(lon2-lon1), sin(lon2-lon1)*cos(lat2))
    baring_degrees = degrees(baring)
    return(baring_degrees)


def select_fluxes_in_radius_of_influence(ceres_lon, ceres_lat, ceres_flux, hst_lon, hst_lat, hst_altitude):
    """
    Selects all of the ceres fluxes that are within the radius of influence of Hubble
    given the altitude of the telescope at that time. Remeber that hst's latitude and lognitude
    MUST be an array of integers of length equal to that of the ceres data points. You can do
    this if, say hst is at (70,80) and length of ceres data is 180, then hst_lat = np.array([70]*180) and
    hst_long = np.array([80]*180).

    BARING is NOT USED in here yet

    Inputs:
    ------
        ceres_lon: (numpy.ndarray) the ceres grid longitudes from netCDF4
        ceres_lat: (numpy.ndarray) the ceres grid latitudes from netCDF4
        ceres_flux: (numpy.ndarray) ceres fluxes (or other attribute) from netCDF4
        hst_lon: (numpy.ndarray) hst longitude array from single point
        hst_lat: (numpy.ndarray) hst latitude from single point
        hst_altitude: (astropy.units.quantity.Quantity) hst altitude in kilometers

    Outputs:
    -------
        result: (numpy.ndarray) an array of all the ceres fluxes that are within the radius of influence
    """

    distance_to_horizon = (calculate_distance_to_horizon(hst_altitude)).value
    result = ceres_flux[vector_calculate_great_circle_distance(
        hst_lon, hst_lat, ceres_lon, ceres_lat) < distance_to_horizon]

    return(result)


def get_fluxes_beneath_hst(ceres_object, hst_lat, hst_lon, hst_altitude):
    """
    Gets all of the fluxes from ceres beneath HST. Uses a loop.... but thats ok for
    now as it is pretty quick (0.35 seconds). Input a ceres field object, like this:

    $ dataframe = Dataset(test_data_path)
    $ flux_dataframe = dataframe.variables['toa_sw_all_daily']
    $ data_flux = flux_dataframe[0, :, :].astype(np.float64)

    This will have to be for the correct time. Here I'm just using the
    first date in the month (0).

    Input:
    ------
        ceres_object: (numpy.ma.core.MaskedArray) the ceres object for all lons and lats and for specified variable
        hst_lat: (int) hst's latitude
        hst_lon: (int) hst's longitude
        hst_altitude: (astropy.units.quantity.Quantity) hst altitude in kilometers

    Output:
    ------
        var_in_region_of_influence: (list) all the fluxes that are in the region of influence
    """
    latitude = np.arange(0, 180)
    longitude = np.arange(0, 360)
    whole_dataframe = []
    for i in latitude:
        longitude_flux_array = ceres_object[i]
        for j in longitude:
            flux = longitude_flux_array[j]
            listout = [i, j, flux]
            whole_dataframe.append(listout)
    return_dataframe = pd.DataFrame.from_records(whole_dataframe, columns=['lat', 'lon', 'flux'])
    var_in_region_of_influence = select_fluxes_in_radius_of_influence(ceres_lon=np.array(list(return_dataframe['lon'])),
                                                                      ceres_lat=np.array(
                                                                          list(return_dataframe['lat'])),
                                                                      ceres_flux=np.array(
        list(return_dataframe['flux'])),
        hst_lon=np.array(
        [hst_lon]*len(return_dataframe['flux'])),
        hst_lat=np.array(
        [hst_lat]*len(return_dataframe['flux'])),
        hst_altitude=hst_altitude)
    return(var_in_region_of_influence)


def old_get_distance_between_two_points(lat1, lon1, lat2, lon2):
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return(R * c)


def get_df_point_hst_can_see(input_data_frame,
                             hst_location_latitude,
                             hst_location_longitude,
                             hst_distance_to_horizon):

    input_data_frame['Distance'] = list(map(lambda k: old_get_distance_between_two_points(
        input_data_frame.loc[k]['Latitude'], input_data_frame.loc[k]['Longitude'], hst_location_latitude, hst_location_longitude), input_data_frame.index))
    new_df = input_data_frame[input_data_frame['Distance'] < hst_distance_to_horizon]
    return(new_df)
