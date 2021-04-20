#
# This file is part of datacube-classification
# Copyright (C) 2021 INPE.
#
# datacube-classification Library is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#
"""datacube-stats cube operations module"""

from typing import List, Dict

import numpy as np
import xarray
from datacube.model import Measurement
from datacube_stats.statistics import Statistic


class Measurements2Cube(Statistic):
    """This statistic class performs the creation of a cube from a specific measurement. This cube is intended for use in
    various products derived from multiple sensors.

    This class was created to meet the need to join cubes with different data types in an easy way, without having to
    join dimensions in a single dataset

    Args:
        measurement (str): measurement name

        measurement_key (str): measurement name in input dataset

        dtype (str): measurement output dtype

        nodata (str): measurement output nodata

        units (str): measurement output units
    """

    def __init__(self, measurement, measurement_key=None, dtype="int16", nodata=255, units="m"):
        self._measurement = measurement
        self._measurement_key = measurement_key or measurement

        self._dtype = dtype
        self._nodata = nodata
        self._units = units

    def compute(self, data: xarray.Dataset) -> xarray.Dataset:
        x, y = np.meshgrid(data.x.values, data.y.values)

        return xarray.Dataset({
            self._measurement: (["x", "y"], data[self._measurement_key].values.reshape((
                data.dims["y"], data.dims["x"]
            ))),
        }, coords={
            "x_coordinate": (["x", "y"], x),
            "y_coordinate": (["x", "y"], y)
        },
            attrs={"crs": data.crs}
        )

    def measurements(self, input_measurements: List[Dict]) -> List:
        return [
            Measurement(name=self._measurement, dtype=self._dtype, nodata=self._nodata, units=self._units),
        ]


def _load_user_defined_function(function_name: str, module_name: str):
    """Function to load arbitrary functions
    Args:
        function_name (str): name of function to load from function_file
        module_name (str): file module where function is defined
    Returns:
        function loaded from file
    """

    import importlib
    return getattr(importlib.import_module(module_name), function_name)


class MeasurementGenerator(Statistic):
    """This statistic class performs the creation of a multi custom measurement cube from user defined functions.
    Args:
        operators (dict): cube measurements specification. This variable must specify all the metadata of the
        new measurement that will be created. An example add a GEMI index is presented below:
            gemi:
              module: datacube_classification.spectral_index
              function: pvr

              dtype: 'int16'
              nodata: -9999
              units: 'm'

              factor: 10000

              args:
                red_band: 'band3'
                green_band: 'band2'

    """

    def __init__(self, operators: dict):
        self._operators: dict = operators

    def compute(self, data: xarray.Dataset) -> xarray.Dataset:
        values = {}

        # apply each user defined function in input data
        for measure in self._operators.keys():
            measure_definition = self._operators[measure]
            measure_function = _load_user_defined_function(measure_definition["function"], measure_definition["module"])

            measure_args = measure_definition["args"]
            measure_factor = measure_definition["factor"]
            values[measure] = (
                ["x", "y"], measure_function(data / measure_factor, **measure_args)[0,] * measure_factor)

        x, y = np.meshgrid(data.x.values, data.y.values)
        return xarray.Dataset(values, coords={
            "x_coordinate": (["x", "y"], x),
            "y_coordinate": (["x", "y"], y)
        }, attrs={
            "crs": data.crs
        })

    def measurements(self, input_measurements: List[Dict]) -> List:
        return [
            Measurement(name=key, dtype=self._operators[key]["dtype"], nodata=self._operators[key]["nodata"],
                        units=self._operators[key]["units"])

            for key in self._operators.keys()
        ]
