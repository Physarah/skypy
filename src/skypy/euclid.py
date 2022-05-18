import re
import urllib
import xmltodict
import warnings
import datetime
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from xml.etree.ElementTree import parse
from xml.etree import ElementTree


class Euclid(object):
    """
    Parse the output of the Euclid background model:
    https://irsa.ipac.caltech.edu/applications/BackgroundModel/docs/dustProgramInterface.html
    Example Usage:
    -------------
    model = Euclid(coordinates = SkyCoord('0h39m15.9s', '0d53m17.016s', frame='icrs'),
                    wavelength = 200*u.micron,
                    date = datetime.datetime(2019, 4, 13),
                    observing_location = 'L2',
                    code_version = 'Wright',
                    median = True)
    model.zodiacal_light
    Inputs:
    ------
       coordinates (astropy.coordinates.sky_coordinate.SkyCoord): coordinates of the field
       wavelength (astropy.units.quantity.Quantity): wavelength of observation
       date (datetime.datetime): date of observation, limited to 2018 to 2029 for L2 position
       observing_location (str): observing location. Either the L2 or Earth
       code_version (str): can be Wright or Kelsall, depending on what model you'd like
       median (bool): if True, will find median zodiacal over a likely viewing range
    Usefull Atributes:
    -----------------
         request (collections.OrderedDict): dictionary of the request output
         zodiacal_light (astropy.units.quantity.Quantity): zodiacal light background (MJy/sr)
         ism (astropy.units.quantity.Quantity): interstellar medium background (MJy/sr)
         stars (astropy.units.quantity.Quantity): stellar background (MJy/sr)
         cib (astropy.units.quantity.Quantity): cosmic infrared background (MJy/sr)
         total_background (astropy.units.quantity.Quantity): total background (MJy/sr).
    """

    def __init__(self, coordinates, wavelength, date, observing_location, code_version, median):

        self.wavelength = (wavelength.to(u.micron)).value

        try:
            transform_coords = coordinates.transform_to('icrs')
            self.locstr = transform_coords.to_string('hmsdms')
        except:
            raise ValueError(
                "{} is not an astropy.coordinates.sky_coordinate.SkyCoord".format(type(coordinates)))

        try:
            self.year, self.day = self.get_year_date(date)
            if self.year < 2018 or self.year > 2029:
                raise ValueError(
                    "This date is not between 2018 and 2029. Please try another date.")
        except:
            pass

        if observing_location == "Earth":
            self.obslocin = '3'
        elif observing_location == "L2":
            self.obslocin = '0'
        else:
            raise ValueError(
                "{} is an incorrect input. Please choose 'Earth' or 'L2'".format(observing_location))

        if code_version == "Wright":
            self.obsverin = '0'

        elif code_version == 'Kelsall':
            self.obsverin = "4"

        else:
            raise ValueError(
                "{} is an incorrect input. Please choose 'Wright' or 'Kelsall'".format(code_version))

        if median:
            self.ido_viewin = '1'
        else:
            self.ido_viewin = '0'

        try:
            self.request = self.send_request(self.parse_url())
            self.request_status = self.request['results']['@status']
            self.zodiacal_light = float((re.findall(
                '\d*\.?\d+', self.request['results']['result']['statistics']['zody'])[0]))*u.megajansky * u.sr**-1
            self.ism = float((re.findall(
                '\d*\.?\d+', self.request['results']['result']['statistics']['ism'])[0]))*u.megajansky * u.sr**-1
            self.stars = float((re.findall(
                '\d*\.?\d+', self.request['results']['result']['statistics']['stars'])[0]))*u.megajansky * u.sr**-1
            self.cib = float((re.findall(
                '\d*\.?\d+', self.request['results']['result']['statistics']['cib'])[0]))*u.megajansky * u.sr**-1
            self.total_background = float((re.findall(
                '\d*\.?\d+', self.request['results']['result']['statistics']['totbg'])[0]))*u.megajansky * u.sr**-1
        except:
            warnings.warn("Sorry, something went wrong. Try again.")

    def parse_url(self):
        base_url = "https://irsa.ipac.caltech.edu/cgi-bin/BackgroundModel/nph-bgmodel?"
        url = base_url + "locstr={}&wavelength={}&year={}&day={}&obslocin={}&ido_viewin={}".format(self.locstr,
                                                                                                   self.wavelength,
                                                                                                   self.year,
                                                                                                   self.day,
                                                                                                   self.obslocin,
                                                                                                   self.obsverin,
                                                                                                   self.ido_viewin)
        return(url.replace(" ", ""))

    def get_year_date(self, date):
        year = date.year
        day = date.timetuple().tm_yday
        return(year, day)

    def send_request(self, url):
        contents = urllib.request.urlopen(url)
        xmldoc = parse(contents)
        tree = xmldoc.getroot()
        xml_str = ElementTree.tostring(tree).decode()
        dictionary_out = xmltodict.parse(xml_str)
        return(dictionary_out)


def get_euclid_prediction(dataset, ints, model="Wright"):
    """use hubble dataset to get euclid sky estimates

    Inputs:
    ------
        dataframe (pandas.core.frame.DataFrame): pandas dataframe like that in /src/data/exampe_data.csv
        ints (int): what row to use to send to euclid observer for sky background estimate

    """
    init_start_times = (dataset["Data Start Time"].tolist())[ints]
    start_times = datetime.datetime.strptime(init_start_times, "%Y-%m-%d %H:%M:%S.%f")
    field_latitude_coordinates = (dataset['GLAT_REF'].tolist())[ints]
    field_longitude_coordinates = (dataset['GLON_REF'].tolist())[ints]
    skycoord = SkyCoord(field_longitude_coordinates,
                        field_latitude_coordinates, unit='deg', frame=Galactic)
    wavelength = ((data_north_goods_F850LP['PHOTPLAM'].tolist())[ints]*u.AA).to(u.micron)
    model = Euclid(location=skycoord,
                   wavelength=wavelength,
                   date=start_times,
                   observing_location='Earth',
                   code_version=model,
                   median=True)
    out = model.zodiacal_light.value
    return(out)


def get_euclid_table(dataframe, obsloc=3, ido_view=0):
    """Instead of sending on request at a time, create a table that can be uploaded to:
    https://irsa.ipac.caltech.edu/applications/BackgroundModel/

    Inputs:
    ------
        dataframe (pandas.core.frame.DataFrame): pandas dataframe like that in /src/data/exampe_data.csv
        obsloc (int): the observer's location "obsloc" is 0 for Earth-Sun L2 and 3 for Earth (defaults to 0)
        ido_view (int): "ido_view" of 1 overrides the day and computes the median value for a typical spacecraft viewing zone (defaults to 1)
    """
    years = []
    days = []
    ras = []
    decs = []
    wavelengths = []
    obslocs = []
    ido_views = []
    for i in np.arange(0, len(dataframe)):
        year, day = get_year_date(datetime.datetime.strptime(
            dataframe['Data Start Time'][i], "%Y-%m-%d %H:%M:%S.%f"))
        field_latitude_coordinates = (dataframe['GLAT_REF'].tolist())[i]
        field_longitude_coordinates = (dataframe['GLON_REF'].tolist())[i]
        skycoord = SkyCoord(field_longitude_coordinates,
                            field_latitude_coordinates, unit='deg', frame=Galactic)
        transform_coords = skycoord.transform_to('icrs')
        coord_ra = transform_coords.ra.deg
        coord_dec = transform_coords.dec.deg
        wavelength = float(((dataframe['PHOTPLAM'].tolist())[i]*u.AA).to(u.micron).value)
        years.append(year)
        days.append(day)
        ras.append(coord_ra)
        decs.append(coord_dec)
        wavelengths.append(wavelength)
        obslocs.append(obsloc)
        ido_views.append(ido_view)

        # to convert to table for Euclid website put above in pandas dataframe df
        # t = Table()
        # t2 = Table.from_pandas(df)
    return(years, days, ras, decs, wavelengths, obslocs, ido_views)
