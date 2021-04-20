#
# This file is part of datacube-classification
# Copyright (C) 2021 INPE.
#
# datacube-classification Library is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#
"""datacube-stats classification operations module"""

import os
from typing import List, Dict

import numpy as np
import xarray
from datacube.model import Measurement
from datacube_stats.statistics import Statistic
from joblib import load

from ..sits import datacube_to_sits


class ScikitLearnClassifier(Statistic):
    """scikit-learn Classifier to be used as datacube-stats Statistics.

    This function loads a pre-trained classifier model from sklearn and uses it to classify the time series
    associated with each pixel associated with the data cube.

    Args:
        classification_model (str): decision tree path model

        factor (int): factor applied to divided data cube values

        quality_band_name (str): name of dimension in `data` where cloud mask is in
    """

    def __init__(self, classification_model: str, quality_band_name: str = None, smoothing: dict = None, factor=10000):
        if not os.path.isfile(classification_model):
            raise RuntimeError("scikit-learn can't be loaded")

        self._factor = factor
        self._quality_band_name = quality_band_name
        self._classification_model = load(classification_model)

        self._smoothing = smoothing

    def compute(self, data: xarray.Dataset) -> xarray.Dataset:
        x, y = np.meshgrid(data.x.values, data.y.values)
        # datacube-stats sometimes generate NA between blocks
        sits = datacube_to_sits(data, quality_band_name=self._quality_band_name, factor=self._factor).fillna(-9999)

        # smooth ?
        if self._smoothing:
            # only bayes is used here
            from ..spatial_smoothing import bayes_spatial_smoothing, guess_type

            classification_probs = (self._classification_model.predict_proba(sits) * self._factor).astype(int)
            classification_probs_smoothed = bayes_spatial_smoothing(classification_probs,
                                                                    xblock_size=data.x.shape[0],
                                                                    yblock_size=data.y.shape[0],
                                                                    **self._smoothing,
                                                                    factor=1 / self._factor)

            classification = guess_type(classification_probs_smoothed)
        else:
            classification = self._classification_model.predict(sits)

        return xarray.Dataset({
            "classification": (["x", "y"], classification.reshape((
                data.dims["y"], data.dims["x"]
            ))),
        }, coords={
            "x_coordinate": (["x", "y"], x),
            "y_coordinate": (["x", "y"], y)
        },
            attrs={"crs": data.crs}
        )

    def measurements(self, input_measurements: List[Dict]) -> List:
        return [Measurement(
            name=f"classification",
            dtype='int16',
            units="m",
            nodata=-9999
        )]
