#
# This file is part of datacube-classification
# Copyright (C) 2021 INPE.
#
# datacube-classification Library is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#
"""classification models module"""

import pandas as pd


def train_sklearn_model(model, labeled_timeseries: pd.DataFrame, label_col="label"):
    """Train a sklearn model using the time series extracted with the
    `datacube_classification.sits.datacube_get_sits` function

    This function receives time series with associated labels and performs the model training. To do this, each
    of the instances present in the input table (labeled_timeseries) must contain a column (label_col) with the
    associated label information

    Args:
        model (object): scikit-learn classification model

        labeled_timeseries (pd.DataFrame): table with time-series extracted from a data cube. Each instance must be have
        a label associated

        label_col (str): column where labels is in `labeled_timeseries`
    Returns:
        object: scikit-learn treined model
    """

    x = labeled_timeseries[labeled_timeseries.columns.difference([label_col])]
    y = labeled_timeseries[label_col].astype(int)

    return model.fit(x, y)
