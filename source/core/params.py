"""Algorithm parameters for the centroid-detection methods.

Frozen dataclasses so the call sites can pass a single value object instead
of a long, easy-to-mismatch keyword list, and so per-dataset configurations
can be defined once and reused across algorithms.

All defaults below are the fine-tuned values from the pre-refactor code at
commit ``ab17d7b`` (the laser-spot dataset under ``images/``). When changing
any default, also update the regression range in ``tests/test_smoke.py``.

Notes on the Gaussian-blur sigmas (FBM + Bessel)
------------------------------------------------
The original code called ``cv2.GaussianBlur(img, (5, 5), 0)`` which lets
OpenCV auto-compute sigma from the kernel size:

    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
          = 0.3 * 1 + 0.8 = 1.1   for ksize = 5

We now go through :func:`source.core.gpu_utils.gaussian_blur` (which uses
``scipy.ndimage.gaussian_filter`` on CPU and ``cupyx.scipy.ndimage`` on GPU),
so sigma is passed explicitly. ``blur_sigma = 1.1`` preserves the original
smoothing strength.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SLICParams:
    n_segments: int = 125
    compactness: float = 10.0
    sigma: float = 5.0


@dataclass(frozen=True)
class FelzenszwalbParams:
    scale: float = 200.0
    sigma: float = 0.5
    min_size: int = 150


@dataclass(frozen=True)
class QuickshiftParams:
    kernel_size: int = 21
    max_dist: float = 50.0
    ratio: float = 5.0


@dataclass(frozen=True)
class FBMParams:
    """OpenCV-moments centroid: blur, threshold, morph close."""
    blur_sigma: float = 1.1     # matches cv2.GaussianBlur(_, (5, 5), 0) default
    threshold: int = 100
    threshold_max: int = 255    # uint8 binary mask ceiling
    morph_kernel: int = 5


@dataclass(frozen=True)
class CCLParams:
    """scikit-image connected-components centroid."""
    blur_sigma: float = 2.0
    morph_disk: int = 5


@dataclass(frozen=True)
class BesselParams:
    """Airy-disk (Bessel) fit.

    ``window_size`` is the half-width of the ROI extracted around the
    initial peak guess. ``initial_blur_sigma`` smooths the image before
    locating that peak. The ``fit_*`` fields seed ``scipy.optimize.curve_fit``:

      p0      = [A0,    x0_init, y0_init, fit_sigma_init, offset_init]
      bounds  = [[0,    x_min,   y_min,   fit_sigma_min,  0           ],
                 [inf,  x_max,   y_max,   inf,            offset_max   ]]
    """
    window_size: int = 80            # half-width of the ROI around the peak
    initial_blur_sigma: float = 1.1  # matches cv2.GaussianBlur(_, (5, 5), 0)
    fit_sigma_init: float = 5.0      # initial guess for the Airy sigma
    fit_sigma_min: float = 0.1       # lower bound for the Airy sigma
    fit_offset_max: float = 1.0      # upper bound for the background offset
