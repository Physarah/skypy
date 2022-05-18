import yaml
import os
import skypy


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
    config = load_config(package_directory('config', 'general'))
    cbds_path = config['PYSYN_CDBS']
    return(cbds_path)
