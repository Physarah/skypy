import warnings
import datetime
import numpy as np
import pandas as pd
import astropy.units as u
import gunagala.sky as skies
import skypy.sky as custom_skies
from gunagala.utils import ensure_unit
from skypy.utils import package_directory
from skypy.telescope import renorm_to_hst_filter
from astropy.time import Time


def get_zodi_SED(scale='none', zodi_norm_filter='acs,wfc1,f606w'):
    """Take Zodi SED from gunagala and scale to
     Hubble F606W using:
     http://www.stsci.edu/hst/wfc3/documents/ISRs/2002/WFC3-2002-12.pdf
     """
    skip_renorm = False
    if isinstance(scale, u.quantity.Quantity):
        if scale.unit == u.erg / u.s / u.cm**2 / \
                u.Angstrom / u.arcsecond**2:
            normunits = 'flam'
        elif scale.unit == u.photon / u.s / u.cm**2 / \
                u.Angstrom / u.arcsecond**2:
            normunits = 'photlam'
        else:
            raise u.UnitsError('Wrong unit of scales provided to get_zoid_SED')
        normalization = scale
    elif scale == 'high':
        normalization = 4.75e-18 * u.erg / u.s / u.cm**2 / \
            u.Angstrom / u.arcsecond**2  # fig1 in PDF
        normunits = 'flam'
    elif scale == 'mid':
        normalization = 3e-18 * u.erg / u.s / u.cm**2 / \
            u.Angstrom / u.arcsecond**2  # fig1 in PDF
        normunits = 'flam'
    elif scale == 'low':
        normalization = 1.72e-18 * u.erg / u.s / u.cm**2 / \
            u.Angstrom / u.arcsecond**2  # NEP values in PDF table
        normunits = 'flam'
    elif scale == 'gunagala':
        # use default gunagala scaling
        skip_renorm = True
    else:
        raise NotImplementedError(
            'Incorrect or not supported scale in get_zodi_SED')

    # load Zodi SED from gunagala
    zodi_gunagala = skies.ZodiacalLight()
    zodi_waves = zodi_gunagala.waves  # um
    zodi_waves_A = zodi_waves.to(u.Angstrom)
    # ph / (arcsec2 m2 s um)
    zodi_photon_sfd_arcsec2 = zodi_gunagala.photon_sfd

    zodi_photon_flam_arcsec2 = zodi_photon_sfd_arcsec2.to(
        u.photon / (u.arcsecond**2 * u.cm**2 * u.s * u.Angstrom))

    if not(skip_renorm):
        if normunits == 'flam':
            ensure_unit(normalization, u.erg /
                        (u.arcsecond**2 * u.cm**2 * u.s * u.Angstrom))
        elif normunits == 'photlam':
            ensure_unit(normalization, u.photon /
                        (u.arcsecond**2 * u.cm**2 * u.s * u.Angstrom))
        else:
            raise u.UnitsError(f'{normunits} not available in pysynphot')
        hubble_zodi_photon_flam_arcsec2_normalised = renorm_to_hst_filter(zodi_photon_flam_arcsec2,
                                                                          zodi_waves_A,
                                                                          zodi_norm_filter,
                                                                          normalization,
                                                                          waveunits='angstrom',
                                                                          fluxunits='photlam',
                                                                          normunits=normunits,
                                                                          surface_brightness=True)

        return(zodi_waves_A, hubble_zodi_photon_flam_arcsec2_normalised)
    else:
        return(zodi_waves_A, zodi_photon_flam_arcsec2)


def get_zodi_SED_with_reddening(red_param, blue_param, scale='none', zodi_norm_filter='acs,wfc1,f606w'):
    """Take Zodi SED from gunagala and scale to
     Hubble F606W using:
     http://www.stsci.edu/hst/wfc3/documents/ISRs/2002/WFC3-2002-12.pdf
     """
    #print("WARNING CONVERT BACK TO OLD SCALING UNITS IN COMMENTS")
    skip_renorm = False
    if isinstance(scale, u.quantity.Quantity):
        if scale.unit == u.erg / u.s / u.cm**2 / \
                u.Angstrom / u.arcsecond**2:
            normunits = 'flam'
        elif scale.unit == u.photon / u.s / u.cm**2 / \
                u.Angstrom / u.arcsecond**2:
            normunits = 'photlam'
        else:
            raise u.UnitsError('Wrong unit of scales provided to get_zoid_SED')
        normalization = scale
    elif scale == 'high':
        normalization = 5.5e-18 * u.erg / u.s / u.cm**2 / \
            u.Angstrom / u.arcsecond**2  # fig1 in PDF 4.75e-18
        normunits = 'flam'
    elif scale == 'mid':
        normalization = 3e-18 * u.erg / u.s / u.cm**2 / \
            u.Angstrom / u.arcsecond**2  # fig1 in PDF 3e-18
        normunits = 'flam'
    elif scale == 'low':
        normalization = 1.5e-18 * u.erg / u.s / u.cm**2 / \
            u.Angstrom / u.arcsecond**2  # NEP values in PDF table 1.72e-18
        normunits = 'flam'
    elif scale == 'gunagala':
        # use default gunagala scaling
        skip_renorm = True
    else:
        raise NotImplementedError(
            'Incorrect or not supported scale in get_zodi_SED')

    # load Zodi SED from gunagala
    zodi_gunagala = custom_skies.ZodiacalLight(_f_blue=blue_param, _f_red=red_param)
    zodi_waves = zodi_gunagala.waves  # um
    zodi_waves_A = zodi_waves.to(u.Angstrom)
    # ph / (arcsec2 m2 s um)
    zodi_photon_sfd_arcsec2 = zodi_gunagala.photon_sfd

    zodi_photon_flam_arcsec2 = zodi_photon_sfd_arcsec2.to(
        u.photon / (u.arcsecond**2 * u.cm**2 * u.s * u.Angstrom))

    if not(skip_renorm):
        if normunits == 'flam':
            ensure_unit(normalization, u.erg /
                        (u.arcsecond**2 * u.cm**2 * u.s * u.Angstrom))
        elif normunits == 'photlam':
            ensure_unit(normalization, u.photon /
                        (u.arcsecond**2 * u.cm**2 * u.s * u.Angstrom))
        else:
            raise u.UnitsError(f'{normunits} not available in pysynphot')
        hubble_zodi_photon_flam_arcsec2_normalised = renorm_to_hst_filter(zodi_photon_flam_arcsec2,
                                                                          zodi_waves_A,
                                                                          zodi_norm_filter,
                                                                          normalization,
                                                                          waveunits='angstrom',
                                                                          fluxunits='photlam',
                                                                          normunits=normunits,
                                                                          surface_brightness=True)

        return(zodi_waves_A, hubble_zodi_photon_flam_arcsec2_normalised)
    else:
        return(zodi_waves_A, zodi_photon_flam_arcsec2)


def get_CIB(delta_wavelength_angstrom=1.,
            normalization=None,
            normalization_filtername='acs,wfc1,f606w',
            get_observations=False):
    # data from:
    # https://www.sciencedirect.com/science/article/pii/S0927650512001752?via%3Dihub#b0610
    # rough pivot wave micron
    # absolute CIB nW m-2 sr-1
    # error CIB nW m-2 sr-1
    # λeff A, from http://svo2.cab.inta-csic.es/svo/theory/fps/index.php
    # bandpass Weff A, from http://svo2.cab.inta-csic.es/svo/theory/fps/index.php
    cib_data = np.array([
        [0.3, 18., 12., 2960.5, 877.3],  # 'HST/WFPC2 B2007'],
        [0.55, 55., 27., 5182.3, 1577.0],  # 'HST/WFPC2 B2007'],
        [0.814, 57., 32., 8186.4, 2501.0],  # 'HST/WFPC2 B2007'],
        [1.25, 54., 17., 1.25 * 10000., (1.471 - 1.086) * 10000],
        [2.2, 28., 7., 2.2 * 10000., (2.458 - 1.977) * 10000]  # 'COBE DIRBE']
    ])

    cib_Wmsr = cib_data[:, 1] * 1e-9 * u.W * u.m**-2 * u.sr**-1  # nW m−2 sr−1
    cib_waves = cib_data[:, 3] * u.Angstrom
    cib_filter_bandpasses = cib_data[:, 4] * u.Angstrom
    cib_sfd_area = (cib_Wmsr / cib_filter_bandpasses).to(
        u.W * u.cm**-2 * u.arcsecond**-2 * u.Angstrom**-1)

    # Need to change units to use spectral_density equivalencies
    cib_unsurface_brightness = cib_sfd_area * u.arcsecond**2
    cib_photon = cib_unsurface_brightness.to(
        u.photon * u.s**-1 * u.cm**-2 * u.Angstrom**-1,
        equivalencies=u.equivalencies.spectral_density(cib_waves))
    # Change back to surface brightness units
    cib_photon_area = cib_photon / u.arcsecond**2

    if get_observations:
        cib_errors_Wmsr = cib_data[:, 2] * 1e-9 * \
            u.W * u.m**-2 * u.sr**-1  # nW m−2 sr−1
        cib_errors_sfd_area = (cib_errors_Wmsr / cib_filter_bandpasses).to(
            u.W * u.cm**-2 * u.arcsecond**-2 * u.Angstrom**-1)
        # Need to change units to use spectral_density equivalencies
        cib_errors_unsurface_brightness = cib_errors_sfd_area * u.arcsecond**2
        cib_errors_photon = cib_errors_unsurface_brightness.to(
            u.photon * u.s**-1 * u.cm**-2 * u.Angstrom**-1,
            equivalencies=u.equivalencies.spectral_density(cib_waves))
        # Change back to surface brightness units
        cib_errors_photon_area = cib_errors_photon / u.arcsecond**2
        return((cib_waves, cib_photon_area, cib_errors_photon_area))

    # interpolate to all wavelengths
    interp_waves = np.arange(min(cib_waves.value), max(
        cib_waves.value), delta_wavelength_angstrom) * u.Angstrom
    interp_photon = np.interp(interp_waves, cib_waves,
                              cib_photon_area) * cib_photon_area.unit

    if normalization is not None:
        ensure_unit(normalization, u.photon /
                    (u.arcsecond**2 * u.cm**2 * u.s * u.Angstrom))
        interp_photon = renorm_to_hst_filter(interp_photon,
                                             interp_waves,
                                             normalization_filtername,
                                             normalization,
                                             surface_brightness=True,
                                             waveunits='angstrom',
                                             fluxunits='photlam',
                                             normunits='photlam')
    return(interp_waves, interp_photon)


def get_OH_emission(scaling, unit):
    oh_data_path = get_package_filename_dir('data', 'OH_line_emission.csv')
    oh_data = pd.read_csv(oh_data_path)
    oh_waves = oh_data['wave']*u.AA
    oh_flux = oh_data['line']*unit
    return(oh_waves, oh_flux)


def get_CIB_lower_limit(delta_wavelength_angstrom=1., get_observations=False):
    """
    Conversion adapted from:
    https://github.com/AstroHuntsman/SkyHopper-2019/blob/master/notebooks/Space%20Eye%20SkyHopper%20comparison.ipynb
    We base this on the lower limits for the integrated galactic light (IGL) from Driver et al 2017.
    There are expressed in terms of total flux for each band so we need to divide by the filter
    bandwidths to convert these to a band averaged spectral flux density, convert to photon based
    units and then interpolate the spectral energy distribution across the Space Eye wavelength
    range.
    """

    driver_path = get_package_filename_dir('data', 'driver_cib.txt')
    driver_data = np.loadtxt(driver_path)
    cib_waves = (driver_data[1, :] * u.micron).to(u.Angstrom)
    cib_filter_bandpasses = (driver_data[3, :] * u.micron).to(u.Angstrom)

    # direct measure, so lower limits, bandpass totals
    cib_limit = driver_data[2, :] * 1e-9 * \
        u.W * u.m**-2 * u.sr**-1  # nW m−2 sr−1
    cib_limit_sfd = (cib_limit / (cib_filter_bandpasses)).to(u.W *
                                                             u.cm**-2 *
                                                             u.arcsecond**-2 *
                                                             u.Angstrom**-1)

    # Need to change units to use spectral_density equivalencies
    cib_unsurface_brightness = cib_limit_sfd * u.arcsecond**2
    cib_limit_photon = cib_unsurface_brightness.to(
        u.photon * u.s**-1 * u.cm**-2 * u.Angstrom**-1,
        equivalencies=u.equivalencies.spectral_density(cib_waves))
    # Change back to surface brightness units
    cib_limit_photon = cib_limit_photon / u.arcsecond**2

    if get_observations:
        return((cib_waves, cib_limit_photon))

    # interpolate to all wavelengths
    interp_waves = np.arange(min(cib_waves.value), max(cib_waves.value),
                             delta_wavelength_angstrom) * u.Angstrom
    interp_limit_photon = np.interp(interp_waves, cib_waves,
                                    cib_limit_photon) * cib_limit_photon.unit

    return(interp_waves, interp_limit_photon)


def get_andrews_cib():
    andrews_path = get_package_filename_dir('data', 'eblmodel_andrews17.csv')
    andrews_data = pd.read_csv(andrews_path)
    andrews_waves = np.array((andrews_data['Wavelength']*10**6)) * u.micron
    andrews_flux = np.array((andrews_data['Total']*10**-9)) * u.Watt * u.m**-2 * u.sr**-1
    normal_units = (andrews_flux.to(u.Watt * u.m**-2 * u.arcsec**-2)*u.arcsec**2)
    final_flux = normal_units.to(u.erg * u.s**-1 * u.cm**-2 * u.Angstrom**-1,
                                 equivalencies=u.equivalencies.spectral_density(andrews_waves))
    return(andrews_waves.to(u.AA), final_flux)


def get_Earthshine(scaling="Low"):
    if scaling == 'Low':
        scale = 1e-20
    # add stuff Sarah
    # I guess if we are going for a more generic earthshine value,
    # we can just scale a TOA spectrum from ceres - not tooooo sure how to scale this

    earth_surface_spectra_path = get_package_filename_dir('data', 'extra_solar.csv')
    earth_surface_spectra = pd.read_csv(earth_surface_spectra_path)

    earth_waves = earth_surface_spectra["Wavelength, microns"]
    earth_waves = earth_waves.values * u.micron
    earthshine_waves = earth_waves.to(u.Angstrom)

    earth_sed = earth_surface_spectra["E-490 W/m2/micron"] * scale
    earthshine = earth_sed.values * u.W * u.m**-2 * u.micron**-1
    earthshine = earthshine.to(u.photon * u.s**-1 * u.cm**-2 * u.Angstrom**-1,
                               equivalencies=u.equivalencies.spectral_density(earthshine_waves))

    # TODO: no idea if this is the right thing to do, it probably isn't!!
    earthshine = earthshine * u.arcsecond**-2
    print('WARNING: need to convert Earthshine to arcsec on sky!!!!')

    return(earthshine_waves, earthshine)


def get_scattered_earthshine_SED(scattering_angle, altitude, scale='none', earth_norm_filter='acs,wfc1,f606w'):

    skip_renorm = False
    if isinstance(scale, u.quantity.Quantity):
        if scale.unit == u.erg / u.s / u.cm**2 / \
                u.Angstrom / u.arcsecond**2:
            normunits = 'flam'
        elif scale.unit == u.photon / u.s / u.cm**2 / \
                u.Angstrom / u.arcsecond**2:
            normunits = 'photlam'
        else:
            raise u.UnitsError('Wrong unit of scales provided to get_zoid_SED')
        normalization = scale
    elif scale == 'high':
        normalization = 4.75e-18 * u.erg / u.s / u.cm**2 / \
            u.Angstrom / u.arcsecond**2  # fig1 in PDF
        normunits = 'flam'
    elif scale == 'mid':
        normalization = 3e-18 * u.erg / u.s / u.cm**2 / \
            u.Angstrom / u.arcsecond**2  # fig1 in PDF
        normunits = 'flam'
    elif scale == 'low':
        normalization = 1.72e-18 * u.erg / u.s / u.cm**2 / \
            u.Angstrom / u.arcsecond**2  # NEP values in PDF table
        normunits = 'flam'
    elif scale == 'gunagala':
        # use default gunagala scaling
        skip_renorm = True
    else:
        raise NotImplementedError(
            'Incorrect or not supported scale in get_zodi_SED')

    earth_surface_spectra_path = get_package_filename_dir('data', 'extra_solar.csv')
    new_earthshine = pd.read_csv(earth_surface_spectra_path)
    waves_e = new_earthshine['Wavelength, microns']
    earth_e = new_earthshine['E-490 W/m2/micron']
    earth_e = (list(earth_e) * u.Watt * u.m**-2 * u.micron**-1).to(u.Watt * u.m**-2 * u.AA**-1)
    waves_e = (list(waves_e) * u.micron).to(u.AA)
    earth_e = earth_e.to(u.photon * u.s**-1 * u.cm**-2 * u.Angstrom**-
                         1, equivalencies=u.spectral_density(waves_e))
    dens = np.exp(-np.mean(altitude)/8.5)
    S = ((np.pi**2*(1.00029**2 - 1)**2)/2) * (dens)/2.504 * \
        10**25 * 1/(waves_e.value**4)*(1+np.cos(scattering_angle)**2)
    stuff = earth_e * S * u.arcsec**-2

    zodi_waves_A = waves_e.to(u.Angstrom)
    # ph / (arcsec2 m2 s um)

    zodi_photon_flam_arcsec2 = stuff

    if not(skip_renorm):
        if normunits == 'flam':
            ensure_unit(normalization, u.erg /
                        (u.arcsecond**2 * u.cm**2 * u.s * u.Angstrom))
        elif normunits == 'photlam':
            ensure_unit(normalization, u.photon /
                        (u.arcsecond**2 * u.cm**2 * u.s * u.Angstrom))
        else:
            raise u.UnitsError(f'{normunits} not available in pysynphot')
        hubble_zodi_photon_flam_arcsec2_normalised = renorm_to_hst_filter(zodi_photon_flam_arcsec2,
                                                                          zodi_waves_A,
                                                                          earth_norm_filter,
                                                                          normalization,
                                                                          waveunits='angstrom',
                                                                          fluxunits='photlam',
                                                                          normunits=normunits,
                                                                          surface_brightness=True)

        return(zodi_waves_A, hubble_zodi_photon_flam_arcsec2_normalised)
    else:
        return(zodi_waves_A, zodi_photon_flam_arcsec2)


def get_rayleigh_Earthshine(scattering_angle, altitude, zodi_norm_filter, normalization):
    if isinstance(normalization, u.quantity.Quantity):
        if normalization.unit == u.erg / u.s / u.cm**2 / \
                u.Angstrom / u.arcsecond**2:
            normunits = 'flam'
        elif normalization.unit == u.photon / u.s / u.cm**2 / \
                u.Angstrom / u.arcsecond**2:
            normunits = 'photlam'
        else:
            raise u.UnitsError('Wrong unit of scales provided to get_zoid_SED')
    earth_surface_spectra_path = get_package_filename_dir('data', 'extra_solar.csv')
    new_earthshine = pd.read_csv(earth_surface_spectra_path)
    waves_e = new_earthshine['Wavelength, microns']
    earth_e = new_earthshine['E-490 W/m2/micron']
    earth_e = (list(earth_e) * u.Watt * u.m**-2 * u.micron**-1).to(u.Watt * u.m**-2 * u.AA**-1)
    waves_e = (list(waves_e) * u.micron).to(u.AA)
    earth_e = earth_e.to(u.photon * u.s**-1 * u.cm**-2 * u.Angstrom**-
                         1, equivalencies=u.spectral_density(waves_e))
    dens = np.exp(-np.mean(altitude)/8.5)
    S = ((np.pi**2*(1.00029**2 - 1)**2)/2) * (dens)/2.504 * \
        10**25 * 1/(waves_e.value**4)*(1+np.cos(scattering_angle)**2)
    stuff = earth_e * S
    if normunits == 'flam':
        ensure_unit(normalization, u.erg /
                    (u.arcsecond**2 * u.cm**2 * u.s * u.Angstrom))
    elif normunits == 'photlam':
        ensure_unit(normalization, u.photon /
                    (u.arcsecond**2 * u.cm**2 * u.s * u.Angstrom))
    else:
        raise u.UnitsError(f'{normunits} not available in pysynphot')
    normalised_rayleigh = renorm_to_hst_filter(stuff,
                                               waves_e,
                                               zodi_norm_filter,
                                               normalization,
                                               waveunits='angstrom',
                                               fluxunits='photlam',
                                               normunits=normunits,
                                               surface_brightness=True)
    return(waves_e, normalised_rayleigh)


def get_blackbody_sed(temp, wavelengths, arcsec=False):
    """
    Calculates a black body spectrum (popping this in for later)

    Parameters
    ----------
    temp: astropy.units.Quantity
                    The temperature of the black body in K
    wavelengths: astropy.units.Quantity
                    Wavelengths to calculate over
    arcsec: bool
        if true, adds in units per arcsec^2
    Returns
    -------
    wavelength : astropy.units.Quantity
                    wavelength array used to calculate fluxes in Amstrong
    flux : astropy.units.Quantity
                    flux from bb in ergs/s/cm^2/AA/arcsec^2

    """
    wavelength = wavelength.to(u.AA)
    bb_model = black_body(temp.to(u.K))
    if arcsec:
        flux = bb_model(wavelength).to(
            FLAM, u.spectral_density(wavelength)) / u.sr
        flux.to(u.erg * u.s**-1 * u.AA ** -1 * u.cm**-2 * u.arcsecond**-2)
        return(wavelength, flux)
    else:
        flux = bb_model(wavelength).to(PHOTLAM, u.spectral_density(wavelength))
        flux.to(u.erg * u.s**-1 * u.AA ** -1 * u.cm**-2)
        return(wavelength, flux)


def sed_input_gunagala(wavelength, sed_waves, sed_flux):
    """
    A callable function not using pysynphot to retrieve a flux as a function of wavelength to interface with gunagala
    No renormalisation capability -  adding this here because of apprehension to use pysynphot.

    Parameters
    ----------
    wavelength: astropy.units.Quantity
        the wavelength in question
    sed_waves: astropy.units.Quantity
        A list of wavelength integers for an SED. Generally comes from seds.py
    sed_flux: astropy.units.Quantity
        A list of fluxes for an SED. Again, generally comes from seds.py.

    Returns
    -------
    needed_flux: astropy.units.Quantity
        A flux corresponding to the wavelength in question

    """
    wavelength = wavelength.to(u.AA)
    needed_flux = np.interp(wavelength, sed_waves.value, sed_flux.value)
    return(needed_flux * u.photon * u.s**-1 * u.cm**-2 * u.AA**-1 * u.arcsec**-2)

    from astropy.coordinates import SkyCoord, Galactic


def get_hst_scaled_zodi_with_redding(date, field, filter_name, blue_param, red_param):

    if isinstance(filter_name, str):
        initialize_zodi = custom_skies.ZodiacalLight(_f_blue=blue_param, _f_red=red_param)
        scaling_factor = initialize_zodi.relative_brightness(field, date)
        scales_dictionary = {'F850LP': 1.32246045*10**-18, "F814W": 1.27*10**-18,
                             "F606W": 1.72*10**-18, "F435W": 1.67*10**-18, "F475W": 1.67*10**-18}
        scale_filter = scales_dictionary[filter_name]
        out_zodi = scaling_factor * scale_filter

    else:
        initialize_zodi = skies.ZodiacalLight()
        scaling_factor = initialize_zodi.relative_brightness(field, date)
        scale_filter = filter_name
        out_zodi = scaling_factor * scale_filter

    return(out_zodi * u.erg * u.s**-1 * u.cm**-2 * u.arcsecond ** -2 * u.Angstrom**-1)


def get_hst_scaled_zodi(date, field, filter_name):
    """
    Qucik ditry function to get the Zodi SED scaled to the brightness in each respective filter.
    These normalisation factors are taken from the NEP and given in http://www.stsci.edu/hst/wfc3/documents/ISRs/2002/WFC3-2002-12.pdf

    Inputs:
    ------
        date: (datetime.datetime) the date from hst object for an observation
        field: (SkyCoord) a skycoord object of the location of the field
        filter_name: (str) the filter that is used in the observation to scale the relative intensity of the zodi
    """
    if isinstance(filter_name, str):
        initialize_zodi = skies.ZodiacalLight()
        scaling_factor = initialize_zodi.relative_brightness(field, date)
        scales_dictionary = {'F850LP': 1.32246045*10**-18, "F814W": 1.27*10**-18,
                             "F606W": 1.72*10**-18, "F435W": 1.67*10**-18, "F475W": 1.67*10**-18}
        scale_filter = scales_dictionary[filter_name]
        out_zodi = scaling_factor * scale_filter

    else:
        initialize_zodi = skies.ZodiacalLight()
        scaling_factor = initialize_zodi.relative_brightness(field, date)
        scale_filter = filter_name
        out_zodi = scaling_factor * scale_filter

    return(out_zodi * u.erg * u.s**-1 * u.cm**-2 * u.arcsecond ** -2 * u.Angstrom**-1)
