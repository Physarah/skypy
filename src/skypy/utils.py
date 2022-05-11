import yaml
import os
import skypy


def load_config(config_path):
    """A general yaml config parser

    Inputs:
    ------
        config_path(str): the path to the config file

    Returns:
    -------
        config_dictionary(dict): a dictionary of the config file
    """
    with open(config_path) as open_file:
        config_dictionary = yaml.safe_load(open_file)
    return(config_dictionary)


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
