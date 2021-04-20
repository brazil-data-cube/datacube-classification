#
# This file is part of datacube-classification
# Copyright (C) 2021 INPE.
#
# datacube-classification Library is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#
"""datacube-stats interpolation operations module"""

from typing import List, Dict

import xarray
from datacube_stats.statistics import Statistic

from ..cloud import cloud_mask
from ..interp import datacube_temporal_interpolate


class TemporalLinearInterpolation(Statistic):
    """scikit-learn Classifier to be used as datacube-stats Statistics.

    This function loads a pre-trained classifier model from sklearn and uses it to classify the time series
    associated with each pixel associated with the data cube.

    Args:
        quality_band_name (str): quality band name (e.g. Fmask4, Cmask)
    """

    def __init__(self, quality_band_name: str):
        self._quality_band_name = quality_band_name

    def compute(self, data: xarray.Dataset) -> xarray.Dataset:
        return datacube_temporal_interpolate(
            cloud_mask(data, self._quality_band_name)
        )

    def measurements(self, input_measurements: List[Dict]) -> List:
        return list(filter(lambda x: x["name"] != self._quality_band_name, input_measurements))
