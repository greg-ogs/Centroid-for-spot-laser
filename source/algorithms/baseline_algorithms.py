"""Bessel Function (Airy disk) fitting for laser-spot centroid detection.

Similar to the module `source.algorithms.centroid_calculator`, this module separates the
actual fit from the visualization.

CUDA acceleration: the heaviest non-fit operation is the initial Gaussian
blur used to locate the peak. That step is delegated to the module `source.core.gpu_utils` and runs on the GPU when
CuPy is available. `scipy.optimize.curve_fit` itself is CPU-only.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import j1
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.util import img_as_float

from source.algorithms.centroid_calculator import save_figure, plot_wireframe
from source.core import gpu_utils
from source.core.params import BesselParams

logger = logging.getLogger(__name__)

PathLike = str | Path

HISTORIC_BESSEL_CSV = "data/historical/bessel-results-images.csv"


def default_output_dir(image_path: PathLike) -> Path:
    """Conventional output directory; mkdir is deferred to save time."""
    return Path("results") / Path(image_path).stem


class BesselFitter:
    """Fits an Airy-disk profile to the brightest spot in an image.

    The model is the diffraction pattern: `I(r) = I_0 (2 J_1(x)/x)^2 + b`.
    """

    WINDOW_SIZE: int = BesselParams().window_size

    def __init__(self, image_path: PathLike, *,
                 params: BesselParams | None = None):
        self.image_path = str(image_path)
        self.params = params or BesselParams()

        image_data = img_as_float(imread(self.image_path))
        if image_data.ndim == 3:
            if image_data.shape[2] == 4:  # RGBA
                image_data = image_data[:, :, :3]
            self.gray_image = rgb2gray(image_data)
        else:
            self.gray_image = image_data

    # --------  Airy disk model --------------------------------------------------------

    @staticmethod
    def airy_disk_2d_function(grid, amplitude, x0, y0, sigma, offset):
        """Vectorised 2-D Airy-disk model used by ``curve_fit``."""
        x, y = grid
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        k = 1.0 / (sigma + 1e-6)
        scaled_r = k * r
        with np.errstate(divide="ignore", invalid="ignore"):
            airy_term = (2 * j1(scaled_r) / scaled_r) ** 2
        airy_term[scaled_r == 0] = 1.0
        return (amplitude * airy_term + offset).ravel()

    # -------- algorithm-only entry point (no plotting) ---------------------

    def fit(self) -> tuple[float, float, np.ndarray]:
        """Return ``(x, y, popt)`` for the fitted centroid. No plotting/I-O."""
        img = self.gray_image
        h, w = img.shape

        # GPU-accelerated Gaussian blur for the initial peak guess.
        blurred = gpu_utils.gaussian_blur(img.astype(np.float32),
                                          sigma=self.params.initial_blur_sigma)
        min_val_ignored, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        min_val = float(blurred.min())
        initial_x, initial_y = max_loc

        ws = self.params.window_size
        y_min = max(0, initial_y - ws)
        y_max = min(h, initial_y + ws)
        x_min = max(0, initial_x - ws)
        x_max = min(w, initial_x + ws)

        roi = img[y_min:y_max, x_min:x_max]
        x_roi, y_roi = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))

        p0 = [max_val - min_val, initial_x, initial_y,
              self.params.fit_sigma_init, min_val]
        bounds = (
            [0, x_min, y_min, self.params.fit_sigma_min, 0],
            [np.inf, x_max, y_max, np.inf, self.params.fit_offset_max],
        )

        try:
            popt, pcov = curve_fit(self.airy_disk_2d_function, (x_roi, y_roi),
                                   roi.ravel(), p0=p0, bounds=bounds)
        except RuntimeError:
            logger.warning("Bessel Fit failed. Reverting to max-intensity location.")
            return float(initial_x), float(initial_y), np.asarray(p0)

        amp, fit_x, fit_y, sigma, offset = popt
        return float(fit_x), float(fit_y), popt

    # -------- Bassel Fit centroid detection (optional rendering) --------------

    def calculate_bessel_centroid(self, *, plot: bool = False,
                                  output_dir: PathLike | None = None) -> tuple[float, float]:
        fit_x, fit_y, popt = self.fit()
        logger.info("Bessel Fit Centroid: X=%.4f, Y=%.4f (Sigma=%.4f)",
                    fit_x, fit_y, popt[3])
        if plot:
            out = Path(output_dir) if output_dir is not None else default_output_dir(self.image_path)
            self.render_roi(fit_x, fit_y, out)
            self.render_results(fit_x, fit_y, popt, out)
        return fit_x, fit_y

    # -------- rendering ----------------------------------------------------

    def render_roi(self, cx: float, cy: float, output_dir: Path) -> None:
        h, w = self.gray_image.shape
        ws = self.params.window_size
        x_min = max(0, int(cx) - ws)
        x_max = min(w, int(cx) + ws)
        y_min = max(0, int(cy) - ws)
        y_max = min(h, int(cy) + ws)
        roi = self.gray_image[y_min:y_max, x_min:x_max]

        plt.rcParams.update({"font.size": 18})
        fig = plt.figure("Bessel ROI", figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(np.rot90(roi), origin="lower", cmap="gray")
        ax.set_title(f"ROI around peak: x[{x_min}:{x_max}], y[{y_min}:{y_max}]")
        ax.set_xlabel("pixels")
        ax.set_ylabel("pixels")
        plt.tight_layout()
        save_figure(fig, output_dir, "Bessel_ROI.png", dpi=200)
        plt.close(fig)

    def render_results(self, cx: float, cy: float, fit_params,
                        output_dir: Path) -> None:
        h, w = self.gray_image.shape

        # 2-D
        plt.rcParams.update({"font.size": 30})
        fig = plt.figure("Bessel Fit Result", figsize=(11, 12.8))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(np.rot90(self.gray_image), origin="lower")
        ax.plot(cy, w - cx, marker="o", markersize=15, color="red",
                label="Bessel Centroid")
        ax.set_xlabel("pixels")
        ax.set_ylabel("pixels")
        ax.legend()
        save_figure(fig, output_dir, "Bessel_2D.png", dpi=200)
        plt.close(fig)

        # 3-D
        plot_wireframe("Bessel", self.gray_image, None, int(cx), int(cy),
                       output_dir, stride=2)

    # -------- batch helper -------------------------------------------------

    @staticmethod
    def process_local_dataset(path: PathLike, *, plot: bool = False) -> str:
        """Run the Bessel fit over every image in `path` and write a CSV."""
        path = str(path)
        image_files = []
        skip_prefixes = ("CCL", "FBM", "Felzenszwalb", "Quickshift", "SLIC", "Bessel")
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith((".png", ".jpg")) and not file.startswith(skip_prefixes):
                    image_files.append(os.path.join(root, file).replace("\\", "/"))

        logger.info("Found %d images to process.", len(image_files))
        results = []
        for image_path in image_files:
            logger.info("Processing: %s", image_path)
            fitter = BesselFitter(image_path)
            start = time.time()
            cx, cy = fitter.calculate_bessel_centroid(plot=plot)
            bessel_time = time.time() - start
            results.append({
                "Image": image_path,
                "Bessel Time": bessel_time,
                "X": cx,
                "Y": cy,
            })
            logger.info("Time: %.6fs   Centroid: X=%.4f, Y=%.4f",
                        bessel_time, cx, cy)

        csv_file = f"bessel-results-{Path(path).name}.csv"
        with open(csv_file, mode="w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["Image", "Bessel Time", "X", "Y"])
            writer.writeheader()
            writer.writerows(results)
        logger.info("Results saved to %s", csv_file)
        return csv_file


class PlottingResults:
    """Loads a Bessel-results CSV and offers basic plotting on top."""

    def __init__(self, csv_path: PathLike):
        self.csv_path = str(csv_path)
        with open(self.csv_path, newline="") as fh:
            self.data = list(csv.DictReader(fh))

    def prepare_data(self) -> pd.DataFrame:
        df = pd.DataFrame(self.data)
        if "Bessel Time" in df.columns:
            df["Bessel Time"] = pd.to_numeric(df["Bessel Time"], errors="coerce")
        return df

    def plot_times(self, output_path: PathLike = "bessel-times-distribution.png",
                   bins: int = 40) -> Path:
        """Histogram of Bessel-fit timings.

        Returns the path the figure was written to. Pure side effect on disk
        plus a closed matplotlib figure (so it is safe to call from scripts
        that also do their own plotting).
        """
        df = self.prepare_data()
        if "Bessel Time" not in df.columns:
            raise ValueError(f"{self.csv_path}: missing 'Bessel Time' column")

        times = df["Bessel Time"].dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(times, bins=bins, color="#3a7ca5", edgecolor="black")
        ax.set_xlabel("Bessel fit time (s)")
        ax.set_ylabel("Count")
        ax.set_title(f"Bessel-fit time distribution (N={len(times)})")
        fig.tight_layout()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Wrote %s", out)
        return out


# Backwards-compat alias for any external import using the old name.
ploting_results = PlottingResults  # noqa: N816 - legacy alias


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"Acceleration: {gpu_utils.device_summary()}")

    if Path(HISTORIC_BESSEL_CSV).exists():
        plotter = PlottingResults(HISTORIC_BESSEL_CSV)
        plotter.plot_times("results/bessel-times-distribution.png")
    else:
        print(f"(no CSV at {HISTORIC_BESSEL_CSV}; "
              "run BesselFitter.process_local_dataset first)")
