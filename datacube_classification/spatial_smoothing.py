#
# This file is part of datacube-classification
# Copyright (C) 2021 INPE.
#
# datacube-classification Library is free software; you can redistribute it and/or modify it
# under the terms of the MIT License; see LICENSE file for more details.
#
"""spatial smoothing models module"""

import numpy as np

import smoothing


def guess_type(cube_arr):
    """Predicts the class based on the highest probability

    Args:
        cube_arr (np.array): cube array data
    """
    return cube_arr.argmax(axis=1)


def bayes_spatial_smoothing(cube_arr, window_dim, xblock_size, yblock_size, factor):
    """Applies class smoothing using a Bayesian smoother

    Args:
        cube_arr (np.array): cube array data

        window_dim (int): spatial smooth window dimension (assume is a square matrix)

        xblock_size (int): Block X size to process data

        yblock_size (int): Block Y size to process data

        factor (number): factor applied in cube array
    """

    cube_arr_shape = cube_arr.shape

    mult_factor = 1 / factor
    window = np.ones((window_dim, window_dim))
    smoothness = np.zeros((cube_arr_shape[1], cube_arr_shape[1]))
    np.fill_diagonal(smoothness, 20)

    # fix probabilities
    maxprob = mult_factor - cube_arr.shape[1] + 1
    cube_arr[cube_arr == 0] = 1
    cube_arr[cube_arr > maxprob] = maxprob

    # compute logit
    logit = np.log(cube_arr / (np.sum(cube_arr, axis=1) - cube_arr.T).T)

    # process Bayesian
    cube_arr = smoothing.bayes_smoother(logit, xblock_size, yblock_size, window, smoothness, False)

    # calculate the Bayesian probability for the pixel
    cube_arr = np.exp(cube_arr) * mult_factor / (np.exp(cube_arr) + 1)
    return cube_arr
