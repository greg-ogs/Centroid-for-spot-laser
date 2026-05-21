"""Centroid-detection algorithms.

Modules
-------
* :mod:`source.algorithms.centroid_calculator` -- SLIC / Felzenszwalb /
  Quickshift / FBM / CCL.
* :mod:`source.algorithms.baseline_algorithms` -- Airy-disk (Bessel) fit.

For convenience, the most commonly used names are re-exported:

    from source.algorithms import Superpixels, BesselFitter
"""
from source.algorithms.baseline_algorithms import BesselFitter, PlottingResults
from source.algorithms.centroid_calculator import (
    Superpixels,
    calculate_centroid,
    calculate_centroid_scikit,
)

__all__ = [
    "Superpixels",
    "calculate_centroid",
    "calculate_centroid_scikit",
    "BesselFitter",
    "PlottingResults",
]
