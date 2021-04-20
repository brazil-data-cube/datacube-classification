#
# This file is part of datacube-classification
# Copyright (C) 2021 INPE.
#
# datacube-classification Library is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#
"""datacube-stats spectral indexes operations module"""

from typing import List, Dict

import xarray
from datacube.model import Measurement
from datacube_stats.statistics import Statistic


class BaseMetrics(Statistic):
    """datacube-stats statistics base class to generate max, min, mean and median metrics
    This class is a base to generate bands metrics (such as max, min, mean, median)
    Args:
        metric_name (str): metric name

        band_name (str): band to be appliend in temporal `metric_name`

        factor (int): factor applied to divided data cube values
    """

    def __init__(self, band_name: str, metric_name: str, factor=10000):
        self._band_name = band_name
        self._metric_name = metric_name
        self._factor = factor

    def compute(self, data: xarray.Dataset) -> xarray.Dataset:
        return xarray.Dataset({
            f"{self._band_name}_{self._metric_name}": getattr(
                getattr(data, self._band_name) / self._factor, self._metric_name)(dim='time') * self._factor
        }, attrs={"crs": data.crs})

    def measurements(self, input_measurements: List[Dict]) -> List:
        return [Measurement(
            name=f"{self._band_name}_{self._metric_name}",
            dtype='int16',
            nodata=-9999,
            units="m"
        )]
