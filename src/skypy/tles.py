import os
import pandas as pd
import datetime
import numpy as np
import warnings
from skypy.utils import (
    get_tle_archive_path, get_tle_storage_path, dump_dictionary_to_yaml)


class TwoLineElementSet(object):
    """
    The class represents a two line element object https://en.wikipedia.org/wiki/Two-line_element_set.
    It graps all the nessesary infomation about the satellite from the tle and makes these into attributes
    that are more human readbale. You can also convert tle dates into datetime objects that make more sense.
    ~ There are no TLE parsers I like :/
    """

    def __init__(self, tle_line_1, tle_line_2):

        self.satellite_catalog_number = tle_line_1[2:7]
        self.classification = tle_line_1[7]
        self.launch_year = tle_line_1[9:11]
        self.launch_day_of_the_year = tle_line_1[11:14]
        self.piece_of_the_launch = tle_line_1[14:17]
        self.epoch_year = tle_line_1[18:20]
        self.epoch_day = tle_line_1[20:32]
        self.ballistic_coefficint_one = tle_line_1[33:43]
        self.ballistic_coefficint_two = tle_line_1[44:52]
        self.radiation_pressure_coefficient = tle_line_1[53:61]
        self.ephemeris_type = tle_line_1[62]
        self.element_set_number = tle_line_1[64:68]
        self.checksum = tle_line_1[68]

        self.inclination = tle_line_2[8:16]
        self.ra_of_assending_node = tle_line_2[17:25]
        self.eccentricity = tle_line_2[26:33]
        self.argument_of_perigee = tle_line_2[34:42]
        self.mean_anomaly = tle_line_2[43:51]
        self.mean_motion = tle_line_2[52:63]
        self.rev_number_at_epoch = tle_line_2[63:68]

    def get_epoch_full_year(self):
        if int(self.epoch_year) > 57:
            # launch of sputnik... yeah, this is going to be a real issue for them in 2057
            epoch_year = str(19) + self.epoch_year
        else:
            epoch_year = str(20) + self.epoch_year
        return(epoch_year)

    def get_epoch_full_date(self):
        full_date = datetime.date(int(self.get_epoch_full_year()), 1, 1) + \
            datetime.timedelta(float(self.epoch_day) - 1)
        return(full_date)


def get_hst_current_tle_from_norad(current_tle_directory='default', norad_url='default'):
    """
    This function quireies NORAD directly and returns the current TLE for HST for the current date and time.
    Unfortunatly this functionality is not avalible for archived TLE's however, which need to be downloaded
    with written permission from the owners of the archive See https://www.celestrak.com/NORAD/documentation/tle-fmt.php
    for reference on how to read the tles.

    Inputs:
    ------
        current_tle_directory: (str) this is where NORAD will save the text file with the infomation
            about HST's orbit. If this is set to 'default' then it will use the directory specified
            in the data_handling.yaml configuration file under the key word 'tle_storage_directory'.
        norad_url: (str) the url for celestrak (see here https://rhodesmill.org/skyfield/earth-satellites.html)
            if set to 'default' it will just use what is given here, however if this url changes you can input your own as a string.

    Outputs:
    -------
        tle_striped_1: (str) this is the first line of the two line element set for hubble as a string
        tle_striped_2: (str) and this is the second line. Both lines are needed to calculate HST empheris infomation

    """
    if current_tle_directory == 'default':
        os.chdir(get_tle_storage_path())
    else:
        os.chdir(current_tle_directory)
    if norad_url == 'default':
        satellites = load.tle('http://celestrak.com/NORAD/elements/science.txt')
    else:
        satellites = load.tle(norad_url)
    hst_satellite = satellites['HST']
    tle_line_1 = open("science.txt", "r").readlines()[4]
    tle_line_2 = open("science.txt", "r").readlines()[5]
    tle_striped_1 = (str(tle_line1)).rstrip("\n\r")
    tle_striped_2 = (str(tle_line2)).rstrip("\n\r")
    return(tle_striped_1, tle_striped_2)


def convert_date_to_nth_day(date_time_object):
    """
    This function takes a datetime object and finds what day of the year is in as an integer

    Inputs:
    ------
        date_time_object: (datetime.datetime) a datetime object to convert to nth day of the year

    Outputs:
    -------
        nth_day: (int) what the number of days in the year have past since that date + the current date.
    """
    new_year_day = pd.Timestamp(year=date_time_object.year, month=1, day=1)
    nth_day = (date_time_object - new_year_day).days + 1
    return(nth_day)


def convert_date_to_tle_format(date_time_object):
    """
    This function takes a datetime object and converts it into a tle format

    Inputs:
    ------
        date_time_object: (datetime.datetime) a datetime object to convert to nth day of the year

    Outputs:
    -------
        constructed_tle_date: (str) the date represented in tle format.
    """
    day_number = convert_date_to_nth_day(date_time_object)
    if len(str(day_number)) == 1:
        day_number = "00" + str(day_number)
    if len(str(day_number)) == 2:
        day_number = "0" + str(day_number)
    if len(str(day_number)) == 3:
        day_number = str(day_number)
    year_input = str(date_time_object.year)
    year_last_two_digits = year_input[-2:]
    constructed_tle_date = str(year_last_two_digits) + str(day_number)
    return(constructed_tle_date)


def calculate_tle_file_length(file_directory):
    """
    Calculate the length of the big text file with the TLE's in it

    Inputs
    -----
        file_directory: (str) the full path to the file in question

    Returns
    ------
        index: (int) the length of the line (i.e the number of rows in it)
    """
    with open(file_directory) as f:
        for i, l in enumerate(f):
            pass
    index = i + 1
    return(index)


def convert_month_date_to_python_integer(datetime_object):
    """
    Because the ceres data structure has time aranges from 0 to X depending on the month, this is just a quick was of getting the int
    of observation from the datetome object.
    """
    day_of_month = datetime_object.day
    python_date = day_of_month - 1
    return(python_date)


def parse_tle_file_to_dictionary(file_name):
    """
    Load the text file of all the tle's supplied, and then load them into a dictionary with a key that makes more sense!
    """
    file_length = calculate_tle_file_length(get_tle_archive_path())
    file_array = np.arange(0, file_length)
    tle_dictionary = {}
    for i in file_array:
        if (i % 2) == 0:  # get every 2 lines in pairs
            line1 = open(get_tle_archive_path(), "r").readlines()[i]
            line2 = open(get_tle_archive_path(), "r").readlines()[i+1]
            tle_striped_1 = (str(line1)).rstrip("\n\r")
            tle_striped_2 = (str(line2)).rstrip("\n\r")
            current_tle = TwoLineElementSet(tle_striped_1, tle_striped_2)
            current_epoch_date = (current_tle.get_epoch_full_date()).isoformat()
            dictionary_addition = {current_epoch_date: [tle_striped_1, tle_striped_2]}
            tle_dictionary.update(dictionary_addition)
    dump_dictionary_to_yaml(file_name, tle_dictionary)
    print("Few! Glad that's over. Btw, you stored the file here: {}".format(
        get_tle_storage_path()+"/"+file_name+".yml"))
    return(tle_dictionary)


def pick_the_right_tle(target_date_time_object, dictionary):
    """
    This function trys to pick a tle that corresponds to the Hubble one quickly. If it can't find one that is
    at least less than 2 days from the original it will warn you and break the loop
    """
    datetime_corrected = target_date_time_object.strftime("%Y-%m-%d")
    try:
        picked_tle = dictionary[datetime_corrected]
        return(picked_tle)
    except KeyError as error:
        try:
            datetime_corrected_plus_1 = target_date_time_object + datetime.timedelta(1)
            target_date_time_object_plus_1 = datetime_corrected_plus_1.strftime("%Y-%m-%d")
            picked_tle = dictionary[target_date_time_object_plus_1]
            return(picked_tle)
        except KeyError as error:
            try:
                datetime_corrected_plus_2 = datetime_corrected_plus_1 + datetime.timedelta(1)
                target_date_time_object_plus_2 = datetime_corrected_plus_1.strftime("%Y-%m-%d")
                picked_tle = dictionary[target_date_time_object_plus_2]
                return(picked_tle)
            except KeyError as error:
                warnings.warn(
                    "I can't find a matching TLE less than 2 days from the requested date. This discrepancy is too large, please find a new TLE")
                warnings.warn("The TLE that failed is {}".format(target_date_time_object))
