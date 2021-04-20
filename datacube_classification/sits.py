#
# This file is part of datacube-classification
# Copyright (C) 2021 INPE.
#
# datacube-classification Library is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#
"""satellite image time series (sits) operations module"""

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray

from datacube_classification.cloud import cloud_mask
from datacube_classification.interp import datacube_temporal_interpolate


def _get_data(datacube, cols, rows, quality_band_name=None) -> pd.DataFrame:
    """Retrieves the time series, for each attribute, associated with the `rows` and `cols` that are being specified.

    This function retrieves and organizes the time series at the attribute level, following the time-first, space-later
    approach. To do this, each of the points specified via the `cols` and `rows` elements
    (Location points in the data coordinate) have the time series retrieved and organized in time.
    This operation is repeated for each attribute. Finally, the elements are arranged in a table where
    each column represents one attribute at one time instant.

    Args:
        datacube (xarray.Dataset): data cube used to extract time series

        cols (list, tuple or np.array): x-axis position

        rows (list, tuple or np.array): y-axis position

        quality_band_name (str): name of dimension in `datacube` where cloud mask is in
    Returns:
        pd.DataFrame: Table with extracted time-series in a attribute-way

    See:
        Land use and cover maps for Mato Grosso State in Brazil from 2001 to 2017  (https://doi.org/10.1038/s41597-020-0371-4)
    """

    output = []

    for col, row in zip(cols, rows):
        data = datacube.sel(x=col, y=row, method="nearest")

        # interpolate!
        if quality_band_name:
            data = datacube_temporal_interpolate(cloud_mask(data, quality_band_name))

        data_bands = list(data.data_vars.keys())
        output.append(
            pd.DataFrame(np.array([
                data[band].values.flatten() for band in data_bands
            ]
            ).ravel())
        )

    index = [
        f"{band}{x}" for band in data_bands for x in range(len(data.time))
    ]
    return pd.concat(output, axis=1).assign(index=index) \
        .set_index("index").T \
        .reset_index(drop=True)


def datacube_get_sits(datacube, geometry_location: gpd.GeoDataFrame, label_col="label", quality_band_name: str = None,
                      factor=10000):
    """Retrieves the time series, for each attribute, associated with the `rows` and `cols` that are being specified.

    This function retrieves the time series for each specified in a GeoDataFrame
    (Required to have the `geometry` column set)

    Args:
        datacube (xarray.Dataset): data cube used to extract time series

        geometry_location (gpd.GeoDataFrame): GeoDataFrame with geometry column. Time-series will be extracted for each
        geometry location

        label_col (str): Column in `geometry_location` where associated label is

        quality_band_name (str): name of dimension in `datacube` where cloud mask is in

        factor (int): factor to be applied in time-series extracted values
    Returns:
        pd.DataFrame: Table with extracted time-series in a attribute-way
    See:
        This function is the same sits_get_data (https://www.rdocumentation.org/packages/sits/versions/1.12.0/topics/sits_get_data)
        in SITS R Package
    """

    geometry_location = geometry_location.copy().to_crs(datacube.crs)
    return (_get_data(datacube, geometry_location.geometry.x, geometry_location.geometry.y,
                      quality_band_name) / factor) \
        .assign(label=geometry_location[label_col])


def datacube_to_sits(datacube, quality_band_name: str = None, factor=10000):
    """Retrieves and organizes the time series associated with all pixels in a data cube.

    This function is optimized for collecting time series associated with all pixels of a data cube. For each pixel
    and its attributes, extractions are made. After that, the data is organized at the attribute level to use in
    the classification process.

    Args:
        datacube (xarray.Dataset): data cube used to extract time series

        factor (int): factor to be applied in time-series extracted values

        quality_band_name (str): name of dimension in `datacube` where cloud mask is in
    Returns:
        pd.DataFrame: Table with extracted time-series in a attribute-way
    """

    # remove cloud shadow (2) and cloud (4) from mask
    if quality_band_name:
        datacube = datacube_temporal_interpolate(
            cloud_mask(datacube, quality_band_name)
        )

    # get dimensions
    xdim = datacube.dims["x"]
    ydim = datacube.dims["y"]

    data_bands = list(datacube.data_vars.keys())
    index = [
        f"{band}{x}"
        for band in data_bands for x in range(len(datacube.time))
    ]

    return pd.concat([
        pd.DataFrame(
            datacube[band].values.reshape(-1, xdim * ydim)
        ) for band in data_bands
    ]).assign(index=index) \
               .set_index("index").T \
               .reset_index(drop=True) / factor


def plot_ts(ts: pd.DataFrame, band_name: str, timeline: list, label_col: str = "label", **kwargs):
    """Plot specific band time series

    Args:
        ts (pd.DataFrame): Table with time series to be plotted (as is in `datacube_get_sits` format)

        band_name (str): band name

        timeline (list): samples timeline

        label_col (str): column in table where labels is in

        kwargs (dict): args to matplotlib.pyplot.figure function
    returns:
        matplotlib.figure.Figure: generated figure
    """
    import matplotlib.pyplot as plt

    labels_groups = ts.groupby(label_col)

    figures = []

    for group_name in labels_groups.groups.keys():
        group_data = labels_groups.get_group(group_name)
        fig = plt.figure(**kwargs)
        ax = plt.gca()

        # create ts plot
        group_data = group_data.filter(like=band_name)
        group_data.columns = pd.to_datetime(timeline)

        group_data = group_data.T
        group_data.plot(legend=False, color='#819bb1', linewidth=0.2, ax=ax)

        group_data.median(axis=1).plot(color='#b16240', ax=ax, linewidth=1.5)
        group_data.quantile(0.25, axis=1).plot(color='#b19540', ax=ax, linewidth=1.5)
        group_data.quantile(0.75, axis=1).plot(color='#b19540', ax=ax, linewidth=1.5)

        plt.title(f"Samples ({group_data.shape[0]}) for class {group_name} in band = {band_name}")
        figures.append(fig)
    return figures
