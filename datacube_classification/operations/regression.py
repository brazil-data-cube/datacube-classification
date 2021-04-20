#
# This file is part of datacube-classification
# Copyright (C) 2021 INPE.
#
# datacube-classification Library is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#
"""datacube-stats regression operations module"""

import os
import shutil
from typing import List, Dict

import xarray
import rioxarray  # used in shadow to export tif from xarray (do not remove!)
import numpy as np
import rasterio as rio

from datacube.model import Measurement
from datacube_stats.statistics import Statistic

import tempfile
import rpy2.robjects
from rpy2.robjects.packages import importr

# R modules
r_utils = importr("utils")
r_raster = importr("raster")
r_RStoolbox = importr("RStoolbox")

r_divide = rpy2.robjects.r["/"]


class SpatioTemporalLinearMixtureModel(Statistic):
    """SpatioTemporal Linear Mixture Model to be used as datacube-stats Statistics. Specific for Landsat-8.

    Args:
        bands (str): list of bands used to generate linear mixture model

        endmembers_file (str): path to csv file with endmembers (each column represent the spectral band and rows
        represents each endmembers)

        factor (int): factor applied to divided data cube values
    See:
        https://www.sciencedirect.com/science/article/abs/pii/S0034425717300500?casa_token=HgXkzGkp2ysAAAAA:gaD0i7DWvbsGS86fNJJ04cAJ-vO7XM-GJAMvEbBs0t6gWArBtPfASvjG5vkXZMfPwWv_TAiky8ii#s0035
    """

    def __init__(self, bands: list, endmembers_file: str, factor=10000):
        self._bands = bands
        self._endmembers = r_utils.read_csv(endmembers_file, header=False)

        self._factor = factor

    def compute(self, data: xarray.Dataset) -> xarray.Dataset:
        stack_bands = []
        tmp_dir = tempfile.mkdtemp()
        mesma_raster_out = os.path.join(tmp_dir, "mesma_raster.tif")

        if data.time.shape[0] > 1:
            raise RuntimeError("This metric works with single time. Before using it, do the temporal composition!")

        for band in self._bands:
            band_path = os.path.join(tmp_dir, f"{band}.tif")

            stack_bands.append(band_path)
            data[band].rio.to_raster(band_path)

        # generate mixture model
        stack_obj = r_raster.stack(stack_bands)
        linear_mixture = r_RStoolbox.mesma(r_divide(stack_obj, self._factor), self._endmembers, method="NNLS")

        r_raster.writeRaster(
            x=linear_mixture,
            filename=mesma_raster_out
        )

        # load fractions and export
        arr = rio.open(mesma_raster_out).read()
        x, y = np.meshgrid(data.x.values, data.y.values)

        shutil.rmtree(tmp_dir)
        return xarray.Dataset({
            "GROUND_FRACTION": (["x", "y"], arr[0, :, :]),
            "VEGETATION_FRACTION": (["x", "y"], arr[1, :, :]),
            "WATER_FRACTION": (["x", "y"], arr[2, :, :])
        },
            coords={
                "x_coordinate": (["x", "y"], x),
                "y_coordinate": (["x", "y"], y)
            }, attrs={"crs": data.crs})

    def measurements(self, input_measurements: List[Dict]) -> List:
        return [
            Measurement(name="GROUND_FRACTION", dtype='float32', nodata=-3.4e+38, units="m"),
            Measurement(name="VEGETATION_FRACTION", dtype='float32', nodata=-3.4e+38, units="m"),
            Measurement(name="WATER_FRACTION", dtype='float32', nodata=-3.4e+38, units="m")
        ]
