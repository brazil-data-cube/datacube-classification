#
# This file is part of datacube-classification
# Copyright (C) 2021 INPE.
#
# datacube-classification Library is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#
"""datacube-stats spectral index module"""

import xarray


def pvr(data: xarray.DataArray, red_band: str, green_band: str):
    """function to generate PVR index
    Args:
        data (xarray.DataArray): data to generate PVR index

        green_band (str): green band name (as in datacube metadata)

        red_band (str): red band name (as in datacube metadata)
    See:
        https://www.indexdatabase.de/db/i-single.php?id=484
    """

    red = data[red_band] / 10000
    green = data[green_band] / 10000

    return (green - red) / (green + red)


def gndvi(data: xarray.DataArray, nir_band: str, green_band: str):
    """function to generate GNDVI index
    Args:
        data (xarray.DataArray): data to generate PVR index

        nir_band (str): nir band name (as in datacube metadata)

        green_band (str): green band name (as in datacube metadata)
    See:
        https://www.indexdatabase.de/db/i-single.php?id=28
    """

    nir = data[nir_band] / 10000
    green = data[green_band] / 10000

    return (nir - green) / (nir + green)


def gemi(data: xarray.DataArray, red_band: str, nir_band: str):
    """function to generate GNDVI index
    Args:
        data (xarray.DataArray): data to generate PVR index

        red_band (str): red band name (as in datacube metadata)

        nir_band (str): nir band name (as in datacube metadata)
    See:
        https://www.indexdatabase.de/db/i-single.php?id=25
    """

    nir = data[nir_band] / 10000
    red = data[red_band] / 10000

    epsilon = (
                      2 * ((nir ** 2) - (red ** 2)) + 1.5 * nir * red
              ) / (nir + red + 0.5)

    return (epsilon * (1 - 0.25 * epsilon)) - ((red - 0.125) / (1 - red))


def ndwi2(data: xarray.DataArray, nir_band: str, green_band: str):
    """function to generate GNDVI index
    Args:
        data (xarray.DataArray): data to generate PVR index

        green_band (str): green band name (as in datacube metadata)

        nir_band (str): nir band name (as in datacube metadata)
    See:
        https://www.indexdatabase.de/db/i-single.php?id=546
    """

    nir = data[nir_band] / 10000
    green = data[green_band] / 10000

    return (green - nir) / (green + nir)
