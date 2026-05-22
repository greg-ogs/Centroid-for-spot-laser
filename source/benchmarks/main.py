"""Process every image under the ``images`` directory and benchmark the
five centroid algorithms (SLIC, Felzenszwalb, Quickshift, FBM, CCL).

Timings reflect *only* the algorithmic cost: matplotlib rendering happens
either not at all (the default) or once at the end, outside the timed
section. Matplotlib pyplot is not thread-safe, so this runner is sequential
unless ``--workers`` is set, in which case work is dispatched through a
process pool.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from source.algorithms import (
    Superpixels,
    calculate_centroid,
    calculate_centroid_scikit,
)
from source.core import gpu_utils

logger = logging.getLogger(__name__)

SKIP_PREFIXES = ("CCL", "FBM", "Felzenszwalb", "Quickshift", "SLIC", "Bessel")
HEADER = ["Image",
          "SLIC Time", "SLIC X", "SLIC Y",
          "Felzenszwalb Time", "Felzenszwalb X", "Felzenszwalb Y",
          "Quickshift Time", "Quickshift X", "Quickshift Y",
          "FBM Time", "FBM X", "FBM Y",
          "CCL Time", "CCL X", "CCL Y"]


def discover_images(path: str) -> list[str]:
    """Find every input image under ``path``."""
    image_files: list[str] = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith((".png", ".jpg")) and not file.startswith(SKIP_PREFIXES):
                image_files.append(os.path.join(root, file).replace("\\", "/"))
    return image_files


def benchmark_image(image_path: str, *, plot: bool = False) -> dict:
    """Time each algorithm on a single image; return a row for the CSV."""
    sp = Superpixels(image_path, 125, 10)

    start = time.time()
    slicX, slicY, _ = sp.calculate_superpixels_slic(plot=plot)
    slic_t = time.time() - start

    start = time.time()
    felzX, felzY, _ = sp.calculate_superpixels_felzenszwalb(plot=plot)
    felz_t = time.time() - start

    start = time.time()
    quickX, quickY, _ = sp.calculate_superpixels_quickshift(plot=plot)
    quick_t = time.time() - start

    start = time.time()
    fbmResult = calculate_centroid(image_path, plot=plot)
    fbm_t = time.time() - start
    fbmX, fbmY = fbmResult if fbmResult is not None else (None, None)

    start = time.time()
    cclResult = calculate_centroid_scikit(image_path, plot=plot)
    ccl_t = time.time() - start
    cclX, cclY = cclResult if cclResult is not None else (None, None)

    return {
        "Image": image_path,
        "SLIC Time": slic_t, "SLIC X": slicX, "SLIC Y": slicY,
        "Felzenszwalb Time": felz_t, "Felzenszwalb X": felzX, "Felzenszwalb Y": felzY,
        "Quickshift Time": quick_t, "Quickshift X": quickX, "Quickshift Y": quickY,
        "FBM Time": fbm_t, "FBM X": fbmX, "FBM Y": fbmY,
        "CCL Time": ccl_t, "CCL X": cclX, "CCL Y": cclY,
    }


def write_csv(results: Iterable[dict], path: str) -> None:
    csv_file = f"results-{Path(path).name}.csv"
    with open(csv_file, mode="w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Results saved to %s", csv_file)


def summarise(results: list[dict], path: str) -> None:
    logger.info("===== SUMMARY =====")
    logger.info("%s: processed %d images", path, len(results))
    if not results:
        return
    keys = ["SLIC Time", "Felzenszwalb Time", "Quickshift Time", "FBM Time", "CCL Time"]
    for k in keys:
        avg = sum(r[k] for r in results) / len(results)
        logger.info("  %-20s: %.6f s", k, avg)


def process_local_dataset(path: str, *, plot: bool = False, workers: int = 1) -> None:
    """Benchmark every image under ``path`` and write a per-folder CSV."""
    image_files = discover_images(path)
    logger.info("[%s] found %d images", path, len(image_files))
    if not image_files:
        return

    results: list[dict] = []
    if workers <= 1:
        for image_path in image_files:
            logger.info("Processing image: %s", image_path)
            results.append(benchmark_image(image_path, plot=plot))
    else:
        # Process pool
        # Each submit() returns a Future (a handle for an in-flight result).
        # as_completed yields them in finish order; .result() blocks and re-raises worker errors.

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(benchmark_image, p, plot=plot): p for p in image_files}
            for fut in as_completed(futures):
                results.append(fut.result())

    write_csv(results, path)
    summarise(results, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="images",
                        help="Root directory containing class subfolders of images")
    parser.add_argument("--plot", action="store_true",
                        help="Render figures alongside the benchmark (slow)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes per dataset folder")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose progress logging (INFO level)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Parses the command-line arguments, configures the logging level, retrieves information
    about device acceleration, and processes local datasets within a specified directory.

    The function initializes logging and processes each subdirectory in the given dataset
    directory. Each subdirectory is processed independently using the provided options such
    as plotting and specifying the number of workers.

    Parameters
    ----------
    argv: Command-line arguments that are typically passed to the script. If None,
                 defaults to sys.argv.
    :type argv: list[str] | None

    Return
    ------
    his function does not return any value; it performs tasks such as logging
             configuration, printing device information, and dataset processing.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )
    print(f"Acceleration: {gpu_utils.device_summary()}")

    dataset_directory = args.dataset
    subdirs = [os.path.join(dataset_directory, d)
               for d in sorted(os.listdir(dataset_directory))
               if os.path.isdir(os.path.join(dataset_directory, d))]

    for actual_path in subdirs:
        process_local_dataset(actual_path, plot=args.plot, workers=args.workers)


if __name__ == "__main__":
    main()
