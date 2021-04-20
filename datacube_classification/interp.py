#
# This file is part of datacube-classification
# Copyright (C) 2021 INPE.
#
# datacube-classification Library is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#
"""data cube interpolation module"""
import xarray


def datacube_temporal_interpolate(datacube, **kwargs):
    """It interpolates the nan values using the temporal dimension.

    This function performs the temporal interpolation of all pixels in a data cube. For its use, it is
    necessary to ensure that all pixels are aligned in time and space.

    Args:
        datacube (xarray.Dataset): data cube used to extract time series
    Returns:
        xarray.Dataset: interpolated data cube
    """

    return datacube.interpolate_na(dim="time", fill_value="extrapolate", **kwargs)
