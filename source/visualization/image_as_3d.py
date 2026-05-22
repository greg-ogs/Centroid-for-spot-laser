"""Render every input image as a 3-D intensity surface.

Used to inspect the raw intensity profile of each laser-spot image
independently of any main pipeline (centroid detection algorithm).

Run from the project root:
    python -m source.visualization.image_as_3d
    python -m source.visualization.image_as_3d --dataset images/c --stride 2
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

logger = logging.getLogger(__name__)

SKIP_PREFIXES = ("CCL", "FBM", "Felzenszwalb", "Quickshift", "SLIC", "Bessel")


class ImageAs3D:
    def __init__(self, image_path: str):
        self.img_path = image_path

    def read_gray(self) -> np.ndarray:
        """Load the image as a grayscale uint8 array."""
        img = ski.io.imread(self.img_path)
        if img.ndim > 2:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = ski.color.rgb2gray(img)
        return ski.util.img_as_ubyte(img)

    @staticmethod
    def render(img: np.ndarray, output_path: Path,
               stride: int = 1, dpi: int = 500) -> Path:
        y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x[::stride, ::stride],
                        y[::stride, ::stride],
                        img[::stride, ::stride],
                        cmap="viridis")

        ax.set_xlabel("X")
        ax.tick_params(axis="both", which="major", labelsize=30)
        ax.set_ylabel("Y")
        ax.set_zlabel("Intensity")
        ax.view_init(elev=10, azim=100)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path


def discover_images(root: str = "images") -> list[str]:
    image_files: list[str] = []
    for r, dirs, files in os.walk(root):
        for file in files:
            if file.endswith((".png", ".jpg")) and not file.startswith(SKIP_PREFIXES):
                image_files.append(os.path.join(r, file).replace("\\", "/"))
    return image_files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="example_images",
                        help="Root directory containing image subfolders")
    parser.add_argument("--out-dir", default="results/3d-airy_examples",
                        help="Directory to write the 3-D surface PNGs")
    parser.add_argument("--stride", type=int, default=1,
                        help="Mesh stride (higher = faster, coarser)")
    parser.add_argument("--dpi", type=int, default=500,
                        help="Output resolution")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose progress logging (INFO level)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    image_files = discover_images(args.dataset)
    logger.info("Found %d images to process.", len(image_files))

    outRoot = Path(args.out_dir)
    for imagePath in image_files:
        im3d = ImageAs3D(imagePath)
        image = im3d.read_gray()
        target = outRoot / f"{Path(imagePath).stem}-3d.png"
        im3d.render(image, target, stride=args.stride, dpi=args.dpi)
        logger.info("  wrote %s", target)


if __name__ == "__main__":
    main()