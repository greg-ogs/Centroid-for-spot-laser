"""Optional CUDA acceleration via CuPy with graceful CPU fallback.

Heavy segmentation algorithms (SLIC, Felzenszwalb, Quickshift) live in
scikit-image and have no GPU implementation, so they always run on CPU.
Image preprocessing (Gaussian blur, threshold, morphological closing,
connected-component labelling, label-wise reductions) can be pushed to
the GPU through CuPy + cupyx.scipy.ndimage when CUDA is available.

Public surface:
    CUDA_AVAILABLE  - bool, True iff cupy imports and a CUDA device is visible
    cp              - the cupy module if available, else None
    cp_ndi          - cupyx.scipy.ndimage if available, else None
    set_force_cpu(force) -> None  (runtime override of CUDA detection)
    get_array_module(use_gpu) -> module (cupy or numpy)
    to_gpu(arr)     -> cupy.ndarray if available, else passthrough
    to_cpu(arr)     -> numpy.ndarray
    gaussian_blur(img, sigma) -> blurred (uses GPU when available)
    label_connected(binary) -> (labels, num)
    label_means(intensity, labels, n_labels) -> mean per label, indexed 0..n
"""
from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

cp = None
cp_ndi = None
CUDA_AVAILABLE: bool = False


def force_cpu_from_env() -> bool:
    return os.environ.get("CENTROID_FORCE_CPU", "").lower() in {"1", "true", "yes"}


def try_initialise_cuda() -> None:
    """Attempt to bind cupy + cupyx and detect a visible CUDA device.

    Updates the module-level ``CUDA_AVAILABLE``, ``cp``, ``cp_ndi``
    in place. Safe to call repeatedly.
    """
    global CUDA_AVAILABLE, cp, cp_ndi
    cp = None
    cp_ndi = None
    CUDA_AVAILABLE = False

    if force_cpu_from_env():
        logger.debug("CENTROID_FORCE_CPU set; staying on CPU.")
        return

    try:
        import cupy as cp_module
        from cupyx.scipy import ndimage as cp_ndi_module

        if cp_module.cuda.runtime.getDeviceCount() > 0:
            cp = cp_module
            cp_ndi = cp_ndi_module
            CUDA_AVAILABLE = True
            logger.info("CUDA enabled via CuPy (device count=%d)",
                        cp_module.cuda.runtime.getDeviceCount())
    except Exception as exc:
        logger.debug("CuPy unavailable, using CPU path: %s", exc)


def set_force_cpu(force: bool) -> None:
    """Toggle the CUDA path at runtime (re-evaluates device availability).

    Use this from tests / configuration code that needs to override the
    import-time decision. Setting ``force=True`` zeroes out cupy bindings
    immediately; ``force=False`` re-runs detection.
    """
    global CUDA_AVAILABLE, cp, cp_ndi
    if force:
        cp = None
        cp_ndi = None
        CUDA_AVAILABLE = False
    else:
        try_initialise_cuda()


# Module-load detection.
try_initialise_cuda()


def get_array_module(use_gpu: bool | None = None):
    """Return the array module to use (cupy or numpy)."""
    if use_gpu is None:
        use_gpu = CUDA_AVAILABLE
    return cp if (use_gpu and CUDA_AVAILABLE) else np


def to_gpu(arr):
    """Move array to GPU if CUDA is available, else return unchanged."""
    if CUDA_AVAILABLE and cp is not None and not isinstance(arr, cp.ndarray):
        return cp.asarray(arr)
    return arr


def to_cpu(arr) -> np.ndarray:
    """Return a numpy.ndarray view of arr (no-op for numpy inputs)."""
    if CUDA_AVAILABLE and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian blur, GPU-accelerated when available.
    """
    if CUDA_AVAILABLE and cp is not None:
        gpu = cp.asarray(img)
        blurred = cp_ndi.gaussian_filter(gpu, sigma=sigma)
        return cp.asnumpy(blurred)
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=sigma)


def label_connected(binary: np.ndarray):
    """Connected-component labelling, GPU-accelerated when available.

    Returns ``(labels, num_features)``.
    """
    if CUDA_AVAILABLE and cp is not None:
        gpu = cp.asarray(binary)
        labels, num = cp_ndi.label(gpu)
        return cp.asnumpy(labels), int(num)
    from scipy.ndimage import label as cpu_label
    labels, num = cpu_label(binary)
    return labels, int(num)


def label_means(intensity: np.ndarray, labels: np.ndarray, n_labels: int) -> np.ndarray:
    """Per-label mean intensity over labels ``0..n_labels``.

    The result has length ``n_labels + 1`` so callers can index by label
    value directly. Labels that do not appear in ``labels`` get NaN; combine
    with ``np.nanargmax`` to pick the brightest existing segment.

    GPU path uses cupyx.scipy.ndimage.mean; CPU path uses scipy.ndimage.mean.
    """
    import warnings

    index = np.arange(0, n_labels + 1)
    if CUDA_AVAILABLE and cp is not None:
        gpu_int = cp.asarray(intensity)
        gpu_lab = cp.asarray(labels)
        gpu_idx = cp.asarray(index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means = cp_ndi.mean(gpu_int, labels=gpu_lab, index=gpu_idx)
        return cp.asnumpy(means)
    from scipy.ndimage import mean as cpu_mean
    with warnings.catch_warnings():
        # scipy warns when a label has zero pixels; we want NaN, no warning.
        warnings.simplefilter("ignore", RuntimeWarning)
        means = cpu_mean(intensity, labels=labels, index=index)
    return np.asarray(means)


def device_summary() -> str:
    if not CUDA_AVAILABLE:
        return "CPU (CuPy unavailable or CENTROID_FORCE_CPU set)"
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        return f"CUDA: {name}"
    except Exception:
        return "CUDA: enabled"