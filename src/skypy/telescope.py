from earthwatch.utils.seds import get_hst_scaled_zodi
from earthwatch.photometry import rate_to_flam, flux_density_to_photon_flux
from earthwatch.utils.hst import get_hst_file_to_parse
from datetime import datetime
import warnings
import ephem
from netCDF4 import Dataset
from astropy.coordinates import Angle, Latitude, Longitude
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy.coordinates import SkyCoord
import re
import pandas as pd
import astropy.units as u
import numpy as np
from os import environ
from astropy.table import Table
from gunagala import camera, optical_filter, optic, psf, sky, imager
from gunagala.utils import ensure_unit
from skypy.utils import load_config, setup_cdbs
S = setup_cdbs()


class HubbleObservation(object):
    """
    HubbleObservation is a class to wrap a hubble data frame.
    It's sort of an easy way of accessing a set of observations, for a particular field
    over many epochs, allowing for easy conversion between different units and so on.
    Mostly, this is designed to be a bandaid for the removal of pysynphot eventually.

    Inputs:
    ------
        observation_dataframe (pandas.core.frame.DataFrame): pandas dataframe like that in /src/data/exampe_data.csv
    """

    def __init__(self, observation_dataframe):

        # string with filter name
        self.observation_dataframe_parsed = observation_dataframe
        # input dataframe
        init_start_times = self.observation_dataframe_parsed["Data Start Time"]
        self.start_times = datetime.strptime(init_start_times, "%Y-%m-%d %H:%M:%S.%f")

        init_end_times = self.observation_dataframe_parsed["Data End Time"]
        self.end_times = datetime.strptime(init_end_times, "%Y-%m-%d %H:%M:%S.%f")

        self.mid_times = self.start_times + (self.end_times - self.start_times)/2

        init_sun_altitudes = self.observation_dataframe_parsed["SUN_ALT"]
        self.sun_altitudes = (init_sun_altitudes) * u.deg
        # the sun altitude as a astropy unit
        init_telescope_altitude = self.observation_dataframe_parsed["ALTITUDE"]
        self.telescope_altitude = (init_telescope_altitude) * u.km
        # altitude of the telescope as a astropy unit
        init_phot_calibration = self.observation_dataframe_parsed["PHOTMODE"]
        new_phot_calibration = re.sub("\s+", ",", init_phot_calibration.strip())
        self.photmode = (new_phot_calibration)
        # Observation configuration for photometric calibration.
        init_sky_background_electrons = self.observation_dataframe_parsed['MDRIZSKY']
        self.sky_background_electrons = init_sky_background_electrons * u.electron
        # a list of sky backgrounds in units of electrons

        self.sky_background_count_rate = self.sky_background_electrons / \
            self.observation_dataframe_parsed['EXPTIME'] * u.second**-1

        init_photflam = self.observation_dataframe_parsed['PHOTFLAM']
        self.photflam = init_photflam * u.erg * \
            u.cm**-2 * u.Angstrom**-1 * u.electron**-1
        # Inverse sensitivity (units: erg cm−2 Å−1 electron−1).
        # This represents the scaling factor necessary to transform an
        # instrumental flux in units of electrons per second to a physical flux density.
        init_photplam = self.observation_dataframe_parsed['PHOTPLAM']
        self.photplam = init_photplam * u.Angstrom
        # Pivot wavelength
        init_photzpt = self.observation_dataframe_parsed['PHOTZPT']
        self.photzpt = init_photzpt
        # STMag zeropoint
        init_exposure_time = self.observation_dataframe_parsed['EXPTIME']
        self.exposure_time = init_exposure_time * u.s
        # exposure times in astropy units
        init_limb_angle = self.observation_dataframe_parsed[
            'Earth Limb Angle Range (deg):LOS_LIMB']
        self.limb_angle = init_limb_angle * u.deg
        # limb angles in astropy units
        init_sun_angle = self.observation_dataframe_parsed['SUNANGLE']
        self.sun_angle = init_sun_angle * u.deg
        # for now, this seems to be the case.

        self.sky_flux_density = rate_to_flam(
            self.sky_background_count_rate, self.photflam)

        self.sky_photon_flux = flux_density_to_photon_flux(self.sky_flux_density, self.photplam)

        self.field_latitude_coordinates = self.observation_dataframe_parsed['GLAT_REF']
        self.field_longitude_coordinates = self.observation_dataframe_parsed['GLON_REF']

        self.skycoord = SkyCoord(self.field_longitude_coordinates,
                                 self.field_latitude_coordinates, unit='deg', frame=Galactic)

        self.filter = self.photmode.split(",")[2]

        self.known_hst_zodi_filters = ['F814W', 'F850LP', 'F606W', 'F435W', 'F475W']
        if any(self.filter in x for x in self.known_hst_zodi_filters):
            self.zodi_strength = get_hst_scaled_zodi(
                self.mid_times, self.skycoord, self.filter)
        else:
            self.zodi_strength = "None"
            warnings.warn('The filter used in the observation is not supported, sorry')

    def calculate_hubble_location(self, tle_line_1, tle_line_2, observation_time, help_bool=False):
        hubble_object = ephem.readtle('HST', tle_line_1, tle_line_2)
        hubble_object.compute(observation_time)
        rad_hubble_longitude = hubble_object.sublong  # Longitude (+E) beneath satellite
        rad_hubble_latitude = hubble_object.sublat  # Latitude (+N) beneath satellite
        if help_bool:
            print(help(rad_hubble_longitude))
            print(help(rad_hubble_latitude))
        deg_hubble_long = np.rad2deg(rad_hubble_longitude)
        deg_hubble_lat = np.rad2deg(rad_hubble_latitude)
        dicts_pos = {"Longitude": deg_hubble_long, "Latitude": deg_hubble_lat}
        return(dicts_pos)


def calc_total_signal_noise(total_countrate,
                            read_noise,
                            dark_current,
                            pixel_area,
                            total_exp_time,
                            num_sub_exposures=1):
    """
    Calculates the signal and noise per pixel.
    Assumes dark current can be perfectly subtracted
    off the signal.

    Parameters
    ----------
    total_countrate : astropy.units.Quantity
        The counts per second of an observation (u.electron/u.s)
    read_noise : astropy.units.Quantity
        Read noise of camera (u.electron)
    dark_current : astropy.units.Quantity
       Dark current of camera (u.electron / u.pixel / u.s)
    gain : astropy.units.Quantity
        Camera gain (u.electrons)
    pixel_area : astropy.units.Quantity
        Total number of pixels in the CCD (u.pixel)
    total_exp_time : astropy.units.Quantity
        Total exposure time (u.s)
    num_sub_exposures : int
        number of sub exposures taken

    Returns
    -------
    signal : astropy.units.Quantity
        Total signal (u.electron)
    noise: astropy.units.Quantity
        Total noise (u.electron)
    """
    # https://gunagala.readthedocs.io/en/develop/_modules/gunagala/imager.html#Imager.extended_source_signal_noise

    signal = total_countrate * total_exp_time  # e
    dark = dark_current * total_exp_time  # e
    read = num_sub_exposures**0.5 * read_noise  # e

    noise = ((signal + dark) * (u.electron / u.pixel) + read**2)**0.5

    signal = signal * pixel_area / u.pixel  # correct for area
    noise = noise * pixel_area**.5

    return(signal, noise)


def calc_hst_signal_noise(count_rate,
                          exp_time,
                          num_sub_exposures=1,
                          filtername='acs,wfc1,fr853n',
                          dark_limit='high'):
    """
    Calulates signal/noise for ACS using F853N (debugging and quick calcs)

    Parameters
    ----------
    count_rate : int
        General (e/pix/s)
    exp_time : int
        Exposure time (s)
    filtername : str
        Filter and detector being used, eg: acs,wfc1,fr853n
    dark_limit : str
        Can be set to 'high' or 'low'. This is the limits of the average dark
        current of the two detectors combined. The same for 'wfc1' or 'wfc2'
    num_sub_exposures : int
        Number of sub exposures used

    Returns
    -------
    signal : astropy.units.Quantity
        Signal for observation (e)
    noise :  astropy.units.Quantity
        Noise for observation (e)
    """

    filter_params = filtername.split(",", 3)
    pixel_area, pixel_scale = get_hst_pixel_area(
        camera=filter_params[0].upper(),
        filtername=filtername,
        masked_fraction=0.5,
        return_pixel_scale=True)
    count_rate_per_pixel = count_rate * pixel_scale**2 * u.pixel
    noise_source = get_hst_noise_parameters(
        filtername=filtername, dark_limit=dark_limit)
    signal, noise = calc_total_signal_noise(total_countrate=count_rate_per_pixel,
                                            read_noise=noise_source['readout'],
                                            dark_current=noise_source['dark'],
                                            pixel_area=pixel_area,
                                            total_exp_time=exp_time,
                                            num_sub_exposures=num_sub_exposures)

    return(signal, noise)


def hst_norm_to_filter(sed, waves, norm_filtername, normalization):
    # https://github.com/numpy/numpy/blob/v1.16.1/numpy/lib/function_base.py#L3982-L4070
    # http://astlib.sourceforge.net/docs/astLib/astLib.astSED-pysrc.html#SED.matchFlux

    fluxes, countrate, waves, hst_area = hst_observe(
        sed, waves, [norm_filtername])

    return((normalization / fluxes.value) * sed.value)


def calc_normalization_hst(sed, waves, norm_filtername, normalization):
    # https://github.com/numpy/numpy/blob/v1.16.1/numpy/lib/function_base.py#L3982-L4070
    # http://astlib.sourceforge.net/docs/astLib/astLib.astSED-pysrc.html#SED.matchFlux

    fluxes, _, _, _ = hst_observe(sed, waves, [norm_filtername])

    return(normalization / fluxes.value)


def make_hst_gunagala_instrument(filternames, no_sky=True):
    """Return an gunagala Imager representing a HST observing mode.

    Args:
        filternames (str list): List  of HST filters.
        no_sky (boolean, optional): If True, it will zero out
        the Zodi sky background in the Hubble instrument instance.

    Returns:
        hubble (gunagala.imager.Imager): An Imager object for the filters
        specified in Args.

    """

    # populate list of gunagala filters from the HST filtername list
    gunagala_filter_list = []
    for current_filtername in filternames:
        filter_params = get_hst_filter_info(current_filtername)
        transmission_table = Table(data=[filter_params['wavelengths'],
                                         filter_params['transmission'] * u.dimensionless_unscaled],
                                   names=['Wavelength', 'Transmission'])
        gunagala_filter = optical_filter.Filter(
            transmission=transmission_table)
        gunagala_filter_list.append(gunagala_filter)
    filters_dict = dict((key, value)
                        for (key, value) in zip(filternames, gunagala_filter_list))

    # prep a perfect transmission as the HST filter already has system throughputs
    all_wavelengths = np.arange(1000, 30000, 1) * u.Angstrom
    perfect_throughput = np.ones(len(all_wavelengths))
    perfect_optical_transmission = Table(data=[all_wavelengths, perfect_throughput * u.dimensionless_unscaled],
                                         names=['Wavelength', 'Throughput'])
    perfect_sensor_QE = Table(data=[all_wavelengths, perfect_throughput * u.electron / u.photon],
                              names=['Wavelength', 'QE'])

    telescope_aperture_area = filter_params['telescope_aperture']
    telescope_aperture_diameter = telescope_aperture_area**0.5
    hst_optics = optic.Optic(aperture=telescope_aperture_diameter,
                             focal_length=(57.6 * u.m),
                             throughput=perfect_optical_transmission)

    # create a gunagala camera instance from the HST/ACS info
    ACS = camera.Camera(bit_depth=16,  # http://www.stsci.edu/hst/acs/documents/handbooks/current/c05_imaging3.html
                        full_well=80000 * u.electron / u.pixel,  # page 5
                        gain=filter_params['gain'],
                        bias=0.03 * u.adu / u.pixel,  # (0.02–0.30%) page 5
                        #  unsure https://pdfs.semanticscholar.org/1a1a/183710dd3274a282d46bdf4bda9426f19e76.pdf
                        readout_time=100 * u.second,
                        #  http://www.stsci.edu/hst/acs/documents/handbooks/cycle19/c03_intro_acs6.html
                        pixel_size=15 * u.micron / u.pixel,
                        resolution=(4096, 2 * 2048) * u.pixel,
                        read_noise=filter_params['readnoise'],
                        dark_current=filter_params['dark_current'],
                        QE=perfect_sensor_QE,
                        minimum_exposure=0.5 * u.second)  # page 158

    # setup PSF
    # http://www.stsci.edu/hst/acs/documents/handbooks/cycle19/c05_imaging7.html#356236
    acs_wfc_psf = psf.MoffatPSF(FWHM=0.13 * u.arcsecond)

    # create final imager using filters, camera, optics, PSF and a temp Zodi sky
    hubble_instrument = imager.Imager(optic=hst_optics,
                                      camera=ACS,
                                      filters=filters_dict,
                                      psf=acs_wfc_psf,
                                      sky=sky.ZodiacalLight(),
                                      num_imagers=1)

    # zero out the sky background if want to measure the sky itself
    if no_sky:
        scaling_factor = 0
        hubble_instrument.sky_rate = {
            filter_name: scaling_factor * sky_rate
            for filter_name, sky_rate in hubble_instrument.sky_rate.items()
        }

    return(hubble_instrument)


def check_hst_units(sed,
                    waves,
                    waveunits='angstrom',
                    fluxunits='photlam',
                    normalization=1 * u.count,
                    normunits=None,
                    surface_brightness=False):

    if surface_brightness:
        surface_brightness = u.arcsecond**-2
    else:
        surface_brightness = 1.

    if fluxunits == 'photlam':
        curr_units_sed = u.photon / \
            (u.cm**2 * u.s * u.Angstrom) * surface_brightness
    else:
        raise NotImplementedError

    if waveunits == 'angstrom':
        curr_units_waves = u.Angstrom
    else:
        raise NotImplementedError

    if normunits == 'flam':
        curr_units_norm = u.erg / u.s / u.cm**2 / u.Angstrom * surface_brightness
    elif normunits == 'photlam':
        curr_units_norm = u.photon / u.s / u.cm**2 / u.Angstrom * surface_brightness
    elif normunits is None:
        curr_units_norm = normalization.unit
    else:
        raise NotImplementedError

    if (sed.unit, waves.unit, normalization.unit) != \
            (curr_units_sed, curr_units_waves, curr_units_norm):
        raise u.UnitsError('ERROR: units to check_hst_units incorrect')

    return(curr_units_sed, curr_units_waves)


def get_hst_filter_width(filternames):
    """Return width of HST filter as list or single astropy quantity in
    units of Angstroms.

    Args:
        filternames (str or str list): List or single string(s) of
        HST filters.

    Returns:
        Quantity or numpy array: array or single Quantity of filter width(s)
        in Units of Angstroms.

    """
    if type(filternames) is str:
        filternames = [filternames]

    widths = []
    for filtername in filternames:
        bp = S.ObsBandpass(filtername)
        width = bp.photbw()
        if len(filternames) == 1:
            return width * u.Angstrom
        widths.append(width * u.Angstrom)
    return(u.Quantity(widths))


def get_hst_area(filternames):
    """Return are HST aperture as list or single astropy quantity in
    units of m^2.

    Args:
        filternames (str or str list): List or single string(s) of
        HST filters.
    Returns:
        Quantity or numpy array: array or single Quantity of HST area
        in Units of metres.
    """
    if type(filternames) is str:
        filternames = [filternames]

    hst_area = (S.ObsBandpass(
        filternames[0]).primary_area * u.cm**2).to(u.m**2)
    return(hst_area)


def get_hst_filter_info(filtername, camera='ACS'):
    """Return dictionary of hst filter info in astropy units.

    Args:
        filtername (str ) eg 'acs,wfc1,fr853n#8400' for ramp filter
        central wavelength must be included.

    Returns:
        Dictionary of the following parameters:
        - dark_current : camera dark current in e/pix/s
        - readnoise : read noise in e/pix
        - field_of_view : pix^2
        - telescope_aperture : m^2
        - gain : e/adu
        - wavelengths : filter wavelengths in Angstroms
        - transmission : filter throughput (dimensionless).
    """

    # get sensor noise parameters
    noise_parameters = get_hst_noise_parameters(filtername,
                                                camera=camera)
    # get HST transmission info
    filter_obs = S.ObsBandpass(filtername)
    wavelengths = filter_obs.wave * u.Angstrom
    throughput = filter_obs.throughput

    # get HST area and FOV
    telescope_aperture = get_hst_area(filtername)
    field_of_view = get_hst_pixel_area(camera=camera, filtername=filtername)

    filter_info = {'dark_current': noise_parameters['dark'],
                   'readnoise': noise_parameters['readout'],
                   'gain': noise_parameters['gain'],
                   'field_of_view': field_of_view,
                   'telescope_aperture': telescope_aperture,
                   'wavelengths': wavelengths,
                   'transmission': throughput}
    return(filter_info)


def get_hst_filter_pivot_wavelength(filternames):
    pivot_waves = []
    for filtername in filternames:
        band = S.ObsBandpass(filtername)
        pivot_waves.append(band.pivot())

    return(pivot_waves)


def hst_observe(sed,
                waves,
                filternames,
                waveunits='angstrom',
                fluxunits='photlam',
                normunits='flam',
                surface_brightness=True):

    curr_units_sed, curr_units_waves = check_hst_units(
        sed, waves, waveunits, fluxunits, surface_brightness=surface_brightness)

    """Push input SED through each HST filter provided."""
    hst_total_sed = S.ArraySpectrum(waves.value, sed.value,
                                    waveunits=waveunits, fluxunits=fluxunits)

    camera = filternames[0].split(',')[0].upper()

    pivot_waves = []
    observed_fluxes = []
    observed_countrate = []
    for filtername in filternames:
        band = S.ObsBandpass(filtername)
        hst_obs = S.Observation(hst_total_sed, band)

        # https://pysynphot.readthedocs.io/en/latest/properties.html#pysynphot-formula-pivwv
        pivot_waves.append(band.pivot())
        # https://pysynphot.readthedocs.io/en/latest/properties.html#pysynphot-formula-effstim
        observed_fluxes.append(hst_obs.effstim(fluxunits))

        # https://pysynphot.readthedocs.io/en/latest/refdata.html#pysynphot-area
        # https://pysynphot.readthedocs.io/en/latest/observation.html#pysynphot-formula-countrate
        observed_countrate.append(hst_obs.countrate())

    if surface_brightness:
        surface_brightness = u.arcsecond**-2
    else:
        surface_brightness = 1.

    hst_area = hst_obs.primary_area * u.cm**2
    observed_fluxes = np.array(observed_fluxes) * \
        curr_units_sed * curr_units_waves * hst_area
    if camera == 'ACS':
        observed_countrate = np.array(
            observed_countrate) * u.electron / u.s * surface_brightness
    else:
        # https://pysynphot.readthedocs.io/en/latest/observation.html
        raise NotImplementedError(
            "{} camera not implemented yet, confirm pysynphot units for countsrate".format(camera))
    pivot_waves = np.array(pivot_waves) * u.Angstrom

    return(observed_fluxes, observed_countrate, pivot_waves, hst_area)


def renorm_to_hst_filter(sed,
                         waves,
                         filtername,
                         normalization,
                         surface_brightness=True,
                         waveunits='angstrom',
                         fluxunits='photlam',
                         normunits='flam'):

    curr_units_sed, curr_units_waves = check_hst_units(sed, waves, waveunits, fluxunits,
                                                       normalization, normunits, surface_brightness)

    # load Zodi SED into pysynphot
    # pysynphot defaults to photlam:
    # https://pysynphot.readthedocs.io/en/latest/units.html
    hubble_sed = S.ArraySpectrum(
        waves.value, sed.value, waveunits='angstrom', fluxunits='photlam')

    # Renormalise to observed Hubble background @ F606W
    # https://pysynphot.readthedocs.io/en/latest/ref_api.html?highlight=renorm
    hubble_sed_normalised = hubble_sed.renorm(
        normalization.value, 'flam', S.ObsBandpass(filtername))
    # add back in units
    hubble_sed_normalised = hubble_sed_normalised.flux * curr_units_sed
    return(hubble_sed_normalised)


def get_hst_noise_parameters(filtername='acs,wfc1,fr853n',
                             dark_limit='high',
                             camera='ACS'):
    """
    Returns noise sources for ACS WFC1 and WFC2 detectors.
    Based upon:
    http://www.stsci.edu/hst/acs/documents/handbooks/current/acs_ihb.pdf

    Parameters
    ----------
    filtername : str
        Name of filter.
    dark_limit : str
        Can be set to 'high' or 'low'. This is the limits of the average dark
        current of the two detectors combined. The same for 'wfc1' or 'wfc2'.
    camera : str
        Name of camera in use (only ACS supported)

    Returns
    -------
    noise_sources : dict
        A dictionary of the noise sources (read, dark) and gain.

    """

    # http://etc.stsci.edu/etcstatic/users_guide/1_ref_10_ccd.html#acs-ccd-parameters
    if camera == 'ACS':
        detector = filtername.split(',')[1]
        gain = 2 * u.electron / u.adu
    else:
        raise NotImplementedError(
            'get_Hubble_noise_parameter doesn\'t work for {} camera'.format(camera))

    if detector == 'wfc1':
        # Amp A read = 4.35
        # Amp B read = 3.73
        read = 4.05 * u.electron / u.pixel
    elif detector == 'wfc2':
        # Amp C read = 4.05
        # Amp D read = 5.05
        read = 4.55 * u.electron / u.pixel

    if dark_limit == 'high':
        dark = 47 * u.electron / u.hour / u.pixel
    elif dark_limit == 'low':
        dark = 37 * u.electron / u.hour / u.pixel
    dark = dark.to(u.electron / u.s / u.pixel)

    noise_params = {'dark': dark,
                    'readout': read,
                    'gain': gain}

    return noise_params


def get_hst_pixel_area(camera='ACS', filtername='acs,wfc1,fr853n', masked_fraction=0.5, return_pixel_scale=False):
    if camera == 'ACS':
        pixel_scale = 0.05 * u.arcsecond / u.pixel

        # http://www.stsci.edu/hst/acs/documents/handbooks/current/c07_obstechniques08.html#155
        # http://www.stsci.edu/hst/acs/documents/handbooks/current/c07_obstechniques08.html#373704
        # http://www.stsci.edu/hst/acs/documents/handbooks/current/c07_obstechniques08.html#370463

        # ACS apertures
        wfc1_iramp_effective_area = 25 * 65 * u.arcsecond**2
        wfc1_mramp_effective_area = 35 * 80 * u.arcsecond**2
        wfc2_oramp_effective_area = 35 * 65 * u.arcsecond**2

        wfc1_effective_area = 202 * 202 * u.arcsecond**2
        wfc2_effective_area = 202 * 202 * u.arcsecond**2

    # remove central wavelength of ramp filter, if it is given
    filtername = filtername.split('#')[0]

    # http://etc.stsci.edu/etcstatic/users_guide/appendix_b_acs.html#ramp-filters
    if filtername == 'acs,wfc1,fr853n':
        effective_area = wfc1_iramp_effective_area
    elif filtername == 'acs,wfc1,fr716n':
        effective_area = wfc1_iramp_effective_area
    elif filtername == 'acs,wfc2,fr782n':
        filtername = wfc2_oramp_effective_area
    elif filtername == 'acs,wfc2,fr931n':
        effective_area = wfc2_oramp_effective_area
    elif filtername == 'acs,wfc1,f606w':
        effective_area = wfc1_effective_area
    else:
        raise NotImplementedError(
            "filter {} not available in get_pixel_area".format(filtername))
    out_area = effective_area / pixel_scale**2
    if return_pixel_scale:
        return (out_area, pixel_scale)
    else:
        return (out_area)
