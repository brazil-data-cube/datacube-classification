#
# This file is part of datacube-classification
# Copyright (C) 2021 INPE.
#
# datacube-classification Library is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#
"""datacube-stats cloud cover operations module"""


def cloud_mask(data, quality_band_name):
    """It clips the data using a cloud mask. The currently supported cloud mask is FMask 4.1,
    where the value `2` represents cloud shadow and the value `4` cloud shadow.

    This function takes an FMask 4.1 cloud mask and applies it to the data so that the pixels where there
    is a cloud have the value NA.

    Args:
        data (xarray.Dataset): xarray dataset with all bands to be masked with cloud mask

        quality_band_name (str): name of dimension in `data` where cloud mask is in
    Returns:
        xarray.Dataset: Dataset masked with cloud mask (without the `quality_band_name` dimension)
    """

    # assume that the quality band was generated using Fmask (v4.1)
    cloud_and_shadow = data[quality_band_name].where(
        (data[quality_band_name] != 2) & (data[quality_band_name] != 4)
    )

    return data.drop(quality_band_name) * cloud_and_shadow.where(cloud_and_shadow.isnull(), 1)
