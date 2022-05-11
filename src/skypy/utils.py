import yaml
import os


def load_config(config_path):
    """a general yaml config parser

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


def package_directory():
    """get the directory where package is installed for data 

    Returns:
    -------
        package_general_path (str): path to installation of package
    """
    package_general_path = os.path.abspath(skypy.__file__)
    return(package_general_path[:-11])
