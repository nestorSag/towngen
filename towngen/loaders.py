import typing as t
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path
import logging

from urllib.request import urlopen
from urllib.error import HTTPError

import pandas as pd
import pycountry
from numpy import array as np_array

logger = logging.getLogger(__name__)

class CityNames:

  """Handles the retrieval of data from a local cache or from the source geographic names database `geonames.org`.
  
  Attributes:
      base_url (str): base URL for downloading data
      cache_path (TYPE): Path to cache downloaded files
      columns (TYPE): Columns of parsed tabular data
      countries (dict): Mapping of country names to codes
  """

  base_url = "http://download.geonames.org/export/zip/{code}.zip"

  columns = [
    "country_code",
    "postal_code",
    "place_name",
    "admin_name1",
    "admin_code1",
    "admin_name2",
    "admin_code2",
    "admin_name3",
    "admin_code3",
    "latitude",
    "longitude",
    "accuracy"
  ]

  cache = ".city_names"

  # list of available countries
  # countries = {country.name.lower(): country.alpha_2 for country in pycountry.countries}

  country_codes = set([country.alpha_2 for country in pycountry.countries])

  @classmethod
  def load(
    cls, 
    country_code: str, 
    query: t.Optional[str] = None,
    raw: bool = False) -> t.Union[np_array, pd.DataFrame]:
    """Retrieves city name data from a given country.
    
    Args:
        country_code (str): Alpha-2 country code, e.g. GB for Great Britain, FR for France.
        query (t.Optional[str], optional): Query string passed to DataFrame instance to filter returned city names. For available columns see class attribute CityNames.columns  or `http://download.geonames.org/export/zip/`.
        raw (bool, optional): If `True`, returns the full downloaded pandas `DataFrame`
    
    Returns:
        np_array: Returns a table with city name data if `raw` is `True`, otherwise returns a numpy array with city names See Handler.columns for the ful list of table columns or have a look at `http://download.geonames.org/export/zip/`.
  
    
    """
    # initialise cache path and list of available country codes
    cache_path = Path(cls.cache)
    cache_path.mkdir(exist_ok=True, parents=True)

    country_code = country_code.upper()
    # sanitise input 
    if country_code not in cls.country_codes:
      raise ValueError(f"'{country_code}' not recognised as country code. See pycountry.countries for available codes.")

    # retrieve from cache or source
    cached_file = cache_path / f"{country_code}.csv"

    if cached_file.is_file():
      data = pd.read_csv(cached_file)
    else:
      logger.info(f"downloading data for country code {country_code}")
      url = cls.base_url.format(code=country_code)

      try:
        resp = urlopen(url)
      except HTTPError as e:
        logger.exception(f"data for '{country_code}' is not available.")

      zipfile = ZipFile(BytesIO(resp.read()))
      byte_data = (zipfile
            .open(f"{country_code}.txt")
            .read())

      data = pd.read_csv(BytesIO(byte_data),sep="\t", header = None, names = cls.columns)

      # save to cache
      data.to_csv(cached_file, index=False)

    data = data.query(query) if query is not None else data

    if raw:
      return data
    else:
      return data["place_name"].unique()

