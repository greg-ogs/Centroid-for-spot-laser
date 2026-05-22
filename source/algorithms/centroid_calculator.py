"""Centroid detection algorithms for laser-spot images.

Public surface:
  * :class:`Superpixels` - SLIC / Felzenszwalb / Quickshift segmentation
    plus the spot-centre pick.
  * :func:`calculate_centroid` - OpenCV moments on the largest contour (FBM).
  * :func:`calculate_centroid_scikit` - scikit-image CCL on the largest region.

Each algorithm accepts a ``plot`` keyword argument; with the default
``plot=False`` the call is suitable for benchmarking. With ``plot=True``
figures are written into ``output_dir`` (defaulting to
``results/<image-stem>/``); the directory is created lazily, only when a
figure is actually saved.

When CUDA is available (via CuPy), preprocessing steps (Gaussian blur,
connected-component labelling, label-wise reductions) are pushed to the GPU
through :mod:`source.core.gpu_utils`. The actual scikit-image segmentation
kernels have no GPU implementation and always run on CPU.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
from skimage.segmentation import felzenszwalb, find_boundaries, mark_boundaries, quickshift, slic
from skimage.util import img_as_float

from source.core import gpu_utils
from source.core.params import (
    CCLParams,
    FBMParams,
    FelzenszwalbParams,
    QuickshiftParams,
    SLICParams,
)

logger = logging.getLogger(__name__)

PathLike = str | Path


# ---------------------------------------------------------------------------
# Output / file helpers
# ---------------------------------------------------------------------------

def default_output_dir(image_path: PathLike) -> Path:
    """Compute the conventional output directory WITHOUT creating it."""
    return Path("results") / Path(image_path).stem


def save_figure(fig, output_dir: Path, filename: str, dpi: int = 150) -> None:
    """Persist ``fig`` to ``output_dir/filename``; mkdir is lazy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Visualization helpers (single render_3d covers surface+wireframe variants)
# ---------------------------------------------------------------------------

def plot_2d(name: str, image_for_display: np.ndarray, x: int, y: int,
             height: int, output_dir: Path) -> None:
    """2-D image with the centroid marker, rotated 90 degrees for display."""
    plt.rcParams.update({"font.size": 30})
    fig = plt.figure(f"Superpixels -- {name}", figsize=(11, 12.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.rot90(image_for_display), origin="lower")
    ax.plot(y, height - x, marker="o", markersize=15, color="red")
    ax.set_xlabel("pixels")
    ax.set_ylabel("pixels")
    save_figure(fig, output_dir, f"{name}.png")
    plt.close(fig)


def render_3d(
    name: str,
    gray_image: np.ndarray,
    segments: np.ndarray | None = None,
    *,
    marker: tuple[int, int] | None = None,
    stride: int = 1,
    surface_alpha: float = 0.7,
    edge_color: str | None = None,
    edge_linewidth: float = 0.0,
    show_boundaries: bool = True,
    view: tuple[float, float] | None = None,
    output_dir: Path,
    suffix: str,
) -> None:
    """Render a 3-D intensity surface, optionally with segment boundaries
    and a centroid marker. Single helper covers both the 'surface' and
    'wireframe' variants used elsewhere in the module.
    """
    rotated_gray = np.rot90(gray_image)
    rotated_segments = np.rot90(segments) if segments is not None else None
    h_rot, w_rot = rotated_gray.shape
    y_grid, x_grid = np.mgrid[0:h_rot, 0:w_rot]

    fig = plt.figure(f"3D Visualization -- {name}", figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    xs = x_grid[::stride, ::stride]
    ys = y_grid[::stride, ::stride]
    zs = rotated_gray[::stride, ::stride] * 256

    if marker is not None:
        mx, my = marker
        ax.plot([my], [h_rot - mx], [258], c="red", marker="o", markersize=17,
                linestyle="None", label="Intensity Peak", zorder=10)

    surface_kwargs = {"cmap": "plasma", "alpha": surface_alpha,
                      "linewidth": edge_linewidth, "antialiased": True,
                      "zorder": 1}
    if edge_color is not None:
        surface_kwargs["edgecolor"] = edge_color
    ax.plot_surface(xs, ys, zs, **surface_kwargs)

    if show_boundaries and rotated_segments is not None:
        boundary = find_boundaries(rotated_segments, mode="outer")
        ys_b, xs_b = np.where(boundary)
        if ys_b.size:
            zs_b = rotated_gray[ys_b, xs_b] * 256 + 2.5
            ax.scatter(xs_b, ys_b, zs_b, c="k", s=0.5, depthshade=False)

    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])
    if marker is not None:
        ax.legend(fontsize=40)

    save_figure(fig, output_dir, f"{name}{suffix}")
    plt.close(fig)


def plot_3d_surface(name: str, gray_image: np.ndarray,
                     segments: np.ndarray | None, output_dir: Path,
                     stride: int = 1) -> None:
    """Plain 3-D intensity surface with optional boundary overlay."""
    render_3d(name, gray_image, segments, stride=stride,
               output_dir=output_dir, suffix="-surface.png")


def plot_wireframe(name: str, gray_image: np.ndarray,
                    segments: np.ndarray | None, x: int, y: int,
                    output_dir: Path, stride: int = 10) -> None:
    """Wireframe-style 3-D surface with centroid marker."""
    render_3d(name, gray_image, segments, marker=(x, y), stride=stride,
               surface_alpha=0.8, edge_color="k", edge_linewidth=0.2,
               view=(40, 250), output_dir=output_dir, suffix="wireframe.png")


def plot_wireframe_legacy(actual_algorithm, rotated_gray_image_meth, x_grid_meth,
                           y_grid_meth, rotated_segments_meth, h_rot_meth, w_rot_meth,
                           x_meth, y_meth, stride=10):
    """Legacy signature kept for external callers (BesselFitter, demos).

    Pre-computed rotated grids are passed in. New code should call
    :func:`plot_wireframe` instead.
    """
    fig = plt.figure(f"3D Visualization -- {actual_algorithm}", figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")

    xs = x_grid_meth[::stride, ::stride]
    ys = y_grid_meth[::stride, ::stride]
    zs = rotated_gray_image_meth[::stride, ::stride] * 256

    ax.plot([y_meth], [h_rot_meth - x_meth], [258], c="red", marker="o",
            markersize=17, linestyle="None", label="Intensity Peak", zorder=10)
    ax.plot_surface(xs, ys, zs, cmap="plasma", alpha=0.8, edgecolor="k",
                    linewidth=0.2, antialiased=True, zorder=1)

    if rotated_segments_meth is not None:
        boundary = find_boundaries(rotated_segments_meth, mode="outer")
        ys_b, xs_b = np.where(boundary)
        if ys_b.size:
            zs_b = rotated_gray_image_meth[ys_b, xs_b] * 256 + 2.5
            ax.scatter(xs_b, ys_b, zs_b, c="k", s=0.5, depthshade=False)

    ax.view_init(elev=40, azim=250)
    ax.legend(fontsize=40)
    fig.savefig(f"{actual_algorithm}wireframe.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Superpixels: SLIC / Felzenszwalb / Quickshift
# ---------------------------------------------------------------------------

class Superpixels:
    """Image-processing utility for superpixel-based centroid detection.

    The image is loaded once and reused across the three segmentation
    strategies. Each ``calculate_*`` method returns ``(x, y, segments)`` and
    only renders figures when ``plot=True``.
    """

    def __init__(self, image_path: PathLike, num_of_segments: int = 110,
                 a_compactness: float = 10, *,
                 output_dir: PathLike | None = None,
                 slic_params: SLICParams | None = None,
                 felzenszwalb_params: FelzenszwalbParams | None = None,
                 quickshift_params: QuickshiftParams | None = None):
        self.image_path = str(image_path)

        # Backwards-compat: accept the old positional args, override defaults.
        self.slic_params = slic_params or SLICParams(n_segments=num_of_segments,
                                                     compactness=a_compactness)
        self.felzenszwalb_params = felzenszwalb_params or FelzenszwalbParams()
        self.quickshift_params = quickshift_params or QuickshiftParams()
        # Kept for any code that read these legacy attributes directly.
        self.n_segments = self.slic_params.n_segments
        self.compactness = self.slic_params.compactness

        image_data = img_as_float(imread(self.image_path))
        if image_data.ndim == 3:
            if image_data.shape[2] == 4:  # RGBA -> RGB
                image_data = image_data[:, :, :3]
            self.gray = rgb2gray(image_data)
        else:
            self.gray = image_data

        # Quickshift requires a colour image; replicate the grey channel.
        self.image_3ch = np.dstack([self.gray] * 3)
        self.height, self.width = self.gray.shape

        self.output_dir = (Path(output_dir) if output_dir is not None
                           else default_output_dir(self.image_path))

        # Backwards-compat for any external code reaching into these attrs.
        self.image_ref = self.image_path
        self.superpixels_images = [self.gray, self.image_3ch]

    # -------- algorithm-only entry points, get the segments for the superpixels algorithms -------------

    def run_slic(self) -> np.ndarray:
        p = self.slic_params
        return slic(self.image_3ch, n_segments=p.n_segments,
                    compactness=p.compactness, sigma=p.sigma)

    def run_felzenszwalb(self) -> np.ndarray:
        p = self.felzenszwalb_params
        return felzenszwalb(self.image_3ch, scale=p.scale, sigma=p.sigma,
                            min_size=p.min_size)

    def run_quickshift(self) -> np.ndarray:
        p = self.quickshift_params
        return quickshift(self.image_3ch, kernel_size=p.kernel_size,
                          max_dist=p.max_dist, ratio=p.ratio)

    # -------- vectorised spot-centre detection ------------------------------

    def center_of_spot(self, segments: np.ndarray,
                       image: np.ndarray | None = None) -> tuple[int, int]:
        """Return the central pixel of the brightest segment.

        Considers labels ``0..max(segments)``; non-existent labels are
        skipped via ``np.nanargmax``. The legacy positional signature
        ``center_of_spot(image, segments)`` is preserved by accepting an
        optional image; if it is 3-D the first channel is used as intensity.
        """
        if image is not None and image is not segments:
            intensity = image[..., 0] if image.ndim == 3 else image
        else:
            intensity = self.gray

        labels_arr = segments
        n_labels = int(labels_arr.max())
        if n_labels < 0:
            return self.width // 2, self.height // 2

        means = gpu_utils.label_means(intensity, labels_arr, n_labels)
        if not np.any(np.isfinite(means)):
            return self.width // 2, self.height // 2
        best = int(np.nanargmax(means))
        ys, xs = np.where(labels_arr == best)
        if xs.size == 0:
            return self.width // 2, self.height // 2
        mid = xs.size // 2
        return int(xs[mid]), int(ys[mid])

    # -------- Getters for the 2 methods required for superpixels final result --------------

    def calculate_superpixels_slic(self, plot: bool = False):
        segments = self.run_slic()
        x, y = self.center_of_spot(segments)
        logger.info("SLIC centroid coordinates are in X=%d, Y=%d", x, y)
        if plot:
            self.render("SLIC", segments, x, y)
        return x, y, segments

    def calculate_superpixels_quickshift(self, plot: bool = False):
        segments = self.run_quickshift()
        x, y = self.center_of_spot(segments)
        logger.info("Quick-shift centroid coordinates are in X=%d, Y=%d", x, y)
        if plot:
            self.render("Quickshift", segments, x, y)
        return x, y, segments

    def calculate_superpixels_felzenszwalb(self, plot: bool = False):
        segments = self.run_felzenszwalb()
        x, y = self.center_of_spot(segments)
        logger.info("Felzenszwalb centroid coordinates are in X=%d, Y=%d", x, y)
        if plot:
            self.render("Felzenszwalb", segments, x, y)
        return x, y, segments

    # -------- rendering -----------------------------------------------------

    def render(self, name: str, segments: np.ndarray, x: int, y: int) -> None:
        marked = mark_boundaries(self.image_3ch, segments)
        plot_2d(name, marked, x, y, self.height, self.output_dir)
        plot_3d_surface(name, self.gray, segments, self.output_dir)
        plot_wireframe(name, self.gray, segments, x, y, self.output_dir)

    # Legacy static helper for callers that pre-compute rotated grids.
    plot_wireframe = staticmethod(plot_wireframe_legacy)


# ---------------------------------------------------------------------------
# FBM (OpenCV moments on the largest contour)
# ---------------------------------------------------------------------------

def calculate_centroid(image_or_path, *, plot: bool = False,
                       output_dir: PathLike | None = None,
                       params: FBMParams | None = None):
    """Centroid of the largest binary blob via OpenCV moments.

    Accepts either a file path or a pre-loaded BGR ``numpy.ndarray``.
    """
    p = params or FBMParams()
    if isinstance(image_or_path, (str, Path)):
        image = cv2.imread(str(image_or_path), cv2.IMREAD_COLOR)
        path_hint = str(image_or_path)
    else:
        image = image_or_path
        path_hint = "fbm-array"
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_or_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = gpu_utils.gaussian_blur(gray, sigma=p.blur_sigma)
    if blurred.dtype != np.uint8:
        blurred = blurred.astype(np.uint8)

    ret, binary = cv2.threshold(blurred, p.threshold, p.threshold_max, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (p.morph_kernel, p.morph_kernel))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("No objects detected!")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = 0, 0
    logger.info("FBM centroid is at (%d, %d)", cx, cy)

    if plot:
        out = Path(output_dir) if output_dir is not None else default_output_dir(path_hint)
        gray_float = gray.astype(np.float32) / 255.0

        result_image = image.copy()
        cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 2)
        cv2.circle(result_image, (cx, cy), 15, (0, 0, 255), -1)
        plt.rcParams.update({"font.size": 30})
        fig = plt.figure("FBM result", figsize=(11, 12.8))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(np.rot90(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)),
                  origin="lower")
        ax.set_xlabel("pixeles")
        ax.set_ylabel("pixeles")
        save_figure(fig, out, "FBM.png")
        plt.close(fig)

        morph_labels = label(morph > 0)
        plot_3d_surface("FBM", gray_float, morph_labels, out)
        plot_wireframe("FBM", gray_float, morph_labels, cx, cy, out, stride=4)

    return cx, cy


# ---------------------------------------------------------------------------
# CCL (scikit-image connected components)
# ---------------------------------------------------------------------------

def calculate_centroid_scikit(image_or_path, *, plot: bool = False,
                              output_dir: PathLike | None = None,
                              params: CCLParams | None = None):
    """Centroid of the largest connected region via scikit-image.

    Accepts either a file path or a pre-loaded ``numpy.ndarray``.
    """
    p = params or CCLParams()
    if isinstance(image_or_path, (str, Path)):
        image = imread(str(image_or_path))
        path_hint = str(image_or_path)
    else:
        image = image_or_path
        path_hint = "ccl-array"

    gray = image if image.ndim == 2 else image[:, :, 0]
    gray_float = img_as_float(gray)

    blurred = gpu_utils.gaussian_blur(gray_float, sigma=p.blur_sigma)
    thresh = threshold_otsu(blurred)
    binary = blurred > thresh
    selem = disk(p.morph_disk)
    morph = closing(binary, selem)

    label_image, nlabels = gpu_utils.label_connected(morph)

    regions = regionprops(label_image)
    if not regions:
        logger.warning("No objects detected!")
        return None

    largest_region = max(regions, key=lambda r: r.area)
    cy_f, cx_f = largest_region.centroid  # skimage uses (row, col)
    cx, cy = int(cx_f), int(cy_f)
    logger.info("CCL centroid is at (%d, %d)", cx, cy)

    if plot:
        out = Path(output_dir) if output_dir is not None else default_output_dir(path_hint)
        plt.rcParams.update({"font.size": 30})
        fig, ax = plt.subplots(figsize=(11, 12.8))
        ax.imshow(np.rot90(image), origin="lower")
        ax.plot(cy, gray.shape[0] - cx, "o", markersize=15, color="red")
        ax.set_xlabel("pixeles")
        ax.set_ylabel("pixeles")
        save_figure(fig, out, "CCL.png")
        plt.close(fig)

        plot_3d_surface("CCL", gray_float, label_image, out)
        plot_wireframe("CCL", gray_float, label_image, cx, cy, out, stride=1)

    return cx, cy


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"Acceleration: {gpu_utils.device_summary()}")
    path_to_image = "images/l0/image100.png"
    sp = Superpixels(path_to_image, 125)

    timings = {}
    for name, fn in [
        ("SLIC", sp.calculate_superpixels_slic),
        ("Felzenszwalb", sp.calculate_superpixels_felzenszwalb),
        ("Quickshift", sp.calculate_superpixels_quickshift),
    ]:
        start = time.time()
        fn(plot=False)
        timings[name] = time.time() - start

    start = time.time()
    calculate_centroid(path_to_image, plot=False)
    timings["FBM"] = time.time() - start

    start = time.time()
    calculate_centroid_scikit(path_to_image, plot=False)
    timings["CCL"] = time.time() - start

    print("\n----- Timings (algorithm only, no rendering) -----")
    for name, t in timings.items():
        print(f"{name}: {t:.6f}s")

    # Render once at the end (not part of the timing) comment out or uncomment depending on the test
    # sp.calculate_superpixels_slic(plot=True)
    # sp.calculate_superpixels_felzenszwalb(plot=True)
    # sp.calculate_superpixels_quickshift(plot=True)
    # calculate_centroid(path_to_image, plot=True)
    # calculate_centroid_scikit(path_to_image, plot=True)
