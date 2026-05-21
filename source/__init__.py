"""Centroid-for-spot-laser source tree.

Layout
------
core           - shared infrastructure (CUDA detection, parameter dataclasses)
algorithms     - centroid detection methods (SLIC/Felz/Quickshift/FBM/CCL, Bessel/Airy fit)
benchmarks     - end-to-end orchestration (timing + CSV output)
visualization  - plotting and format-conversion utilities
"""
