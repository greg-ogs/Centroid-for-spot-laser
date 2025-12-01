import csv
import os
import time
from threading import Thread

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import j1
from skimage.io import imread
from skimage.util import img_as_float
from skimage.color import rgb2gray


class BesselFitter:
    """
    Represents an image processing utility for calculating centroids using
    Bessel Function Fitting (Airy Disk approximation).

    This technique is ideal for optical spots where the intensity distribution
    follows a diffraction pattern defined by:
    I(r) = Io * (2 * J1(x) / x)^2 + Background
    """

    def __init__(self, image_path):
        """
        Initializes the BesselFitter with an image path.

        :param image_path: Path to the input image.
        :type image_path: str
        """
        self.image_path = image_path

        # Load and preprocess image similar to Superpixels class
        image_data = img_as_float(imread(image_path))

        # Ensure we have a 2D grayscale image
        if image_data.ndim == 2:
            self.gray_image = image_data
        else:
            if image_data.shape[2] == 4:  # Handle RGBA
                image_data = image_data[:, :, :3]
            self.gray_image = rgb2gray(image_data)

    @staticmethod
    def airy_disk_2d_function(grid, amplitude, x0, y0, sigma, offset):
        """
        The 2D Airy Disk function model for fitting.

        Z = Amplitude * [ 2*J1(k*r) / (k*r) ]^2 + Offset
        """
        x, y = grid
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

        # Avoid division by zero at the exact center (limit approaches 1)
        # k is represented implicitly by 1/sigma here for fitting stability
        k = 1.0 / (sigma + 1e-6)
        scaled_r = k * r

        # Calculate the Airy term. Handle r=0 case where function is 1.
        with np.errstate(divide='ignore', invalid='ignore'):
            airy_term = (2 * j1(scaled_r) / scaled_r) ** 2
        airy_term[scaled_r == 0] = 1.0

        return (amplitude * airy_term + offset).ravel()

    def calculate_bessel_centroid(self):
        """
        Performs the Bessel Function fit on the loaded image to find the centroid.

        It locates the brightest pixel as an initial guess, extracts a Region of
        Interest (ROI) to improve fitting speed, and applies non-linear least squares.

        :return: (x, y) coordinates of the centroid.
        """
        img = self.gray_image
        h, w = img.shape

        # 1. Initial Guess based on max intensity
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)

        initial_x, initial_y = max_loc

        # 2. Define Region of Interest (ROI) for optimization
        # Crop a window around the max spot.
        window_size = 80  # +/- pixels around center
        y_min = max(0, initial_y - window_size)
        y_max = min(h, initial_y + window_size)
        x_min = max(0, initial_x - window_size)
        x_max = min(w, initial_x + window_size)

        roi = img[y_min:y_max, x_min:x_max]

        # Create grid for the ROI
        x_roi, y_roi = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))

        # --- ROI Visualization (for debugging/inspection) ---
        try:
            plt.rcParams.update({'font.size': 18})
            fig_roi = plt.figure("Bessel ROI", figsize=(6, 6))
            ax_roi = fig_roi.add_subplot(1, 1, 1)

            # Keep orientation consistent with other visualizations (rotated)
            ax_roi.imshow(np.rot90(roi), origin='lower', cmap='gray')
            ax_roi.set_title(f"ROI around peak: x[{x_min}:{x_max}], y[{y_min}:{y_max}]")
            ax_roi.set_xlabel("pixels")
            ax_roi.set_ylabel("pixels")
            plt.tight_layout()
            plt.savefig('Bessel_ROI.png', dpi=200)
            plt.show()
        except Exception as e:
            # Non-fatal: continue if running headless or any display error occurs
            print(f"Warning: Unable to display/save ROI image: {e}")

        # 3. Curve Fit
        # Parameters: [Amplitude, x0, y0, sigma (width), offset]
        p0 = [max_val - min_val, initial_x, initial_y, 5.0, min_val]

        # Bounds to ensure the centroid stays within the image and physics make sense
        bounds = (
            [0, x_min, y_min, 0.1, 0],  # Lower bounds
            [np.inf, x_max, y_max, np.inf, 1.0]  # Upper bounds
        )

        try:
            popt, pcov = curve_fit(self.airy_disk_2d_function, (x_roi, y_roi), roi.ravel(), p0=p0, bounds=bounds)
        except RuntimeError:
            print("Bessel Fit failed. Reverting to Max intensity location.")
            return initial_x, initial_y

        fit_amp, fit_x, fit_y, fit_sigma, fit_offset = popt

        print(f"Bessel Fit Centroid: X={fit_x:.4f}, Y={fit_y:.4f} (Sigma={fit_sigma:.4f})")

        # 4. Visualization
        self.plot_results(fit_x, fit_y, popt)

        return fit_x, fit_y

    def plot_results(self, cx, cy, fit_params):
        """
        Visualizes the results including a 2D heatmap with marker and a 3D wireframe
        comparing the raw data vs the fitted Bessel function.
        """
        # --- 2D Visualization ---
        plt.rcParams.update({'font.size': 30})
        fig = plt.figure("Bessel Fit Result", figsize=(11, 12.8))
        ax = fig.add_subplot(1, 1, 1)

        # Rotated 90 degrees to match your existing coordinate convention
        ax.imshow(np.rot90(self.gray_image), origin='lower')

        # Plot marker (Note: Y and X swapped and inverted based on your original code logic)
        plt.plot(cy, 1280 - cx, marker='o', markersize=15, color='red', label='Bessel Centroid')

        plt.xlabel("pixels")
        plt.ylabel("pixels")
        plt.legend()
        plt.axis("on")
        plt.savefig('Bessel_2D.png', dpi=200)
        plt.show()

        # --- 3D Visualization ---
        # Generate the fitted surface data for the whole image (or a larger ROI for viz)
        h, w = self.gray_image.shape

        # Rotate image for consistency with 2D plot
        rotated_gray_image = np.rot90(self.gray_image)
        h_rot, w_rot = rotated_gray_image.shape
        y_grid, x_grid = np.mgrid[0:h_rot, 0:w_rot]

        # Calculate fitted surface values
        # Note: We must map the rotated grid back to original coordinates to calculate Z
        # Original X corresponds to current Y grid, Original Y corresponds to (Height - current X grid)
        # This mapping depends heavily on exactly how np.rot90 interacts with your camera coordinates.
        # Assuming standard rotation:

        # For visualization, we will plot the RAW 3D surface and the Centroid Point
        fig_3d = plt.figure("3D Visualization -- Bessel", figsize=(15, 15))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        stride = 2  # Optimization for rendering speed
        xs = x_grid[::stride, ::stride]
        ys = y_grid[::stride, ::stride]
        zs = rotated_gray_image[::stride, ::stride] * 256

        # Plot Intensity Peak
        # Using the coordinate transform found in your Superpixels.plot_wireframe
        ax_3d.plot([cy], [1280 - cx], [258], c='red', marker='o', markersize=17,
                   linestyle='None', label='Bessel Centroid', zorder=10)

        # Plot raw data surface
        ax_3d.plot_surface(xs, ys, zs, cmap='plasma', alpha=0.6, edgecolor='k',
                           linewidth=0.1, antialiased=True, zorder=1)

        # Optional: Plot the Fitted Bessel Function as a wireframe mesh on top
        # (Commented out to prevent visual clutter, but available if needed)
        # amp, x0, y0, sig, off = fit_params
        # z_fit = self.airy_disk_2d_function((ys, 1280-xs), amp, x0, y0, sig, off).reshape(xs.shape) * 256
        # ax_3d.plot_wireframe(xs, ys, z_fit, color='cyan', alpha=0.5, rstride=5, cstride=5)

        ax_3d.view_init(elev=40, azim=250)
        ax_3d.legend(fontsize=30)
        plt.savefig('Bessel_3D_Wireframe.png', dpi=200)
        plt.show()

    @staticmethod
    def process_local_dataset(path):
        image_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".png") or file.endswith(".jpg"):
                    # Skip result images (those that are named after algorithms)
                    if not any(file.startswith(alg_result) for alg_result in
                               ["CCL", "FBM", "Felzenszwalb", "Quickshift", "SLIC"]):
                        # Use forward slashes for compatibility with the existing code
                        file_path = os.path.join(root, file).replace("\\", "/")
                        image_files.append(file_path)

        print(f"Found {len(image_files)} images to process:")
        for img in image_files:
            print(f"- {img}")

        # Process each image
        results = []
        for path_to_image in image_files:
            print(f"\nProcessing image: {path_to_image}")

            # Initialize Fitter
            bessel_tool = BesselFitter(path_to_image)

            # Calculate
            start = time.time()
            cx, cy = bessel_tool.calculate_bessel_centroid()
            end = time.time()

            bessel_time = end - start

            result = {
                'Image': path_to_image,
                'Bessel Time': bessel_time
            }
            results.append(result)

            print("\n===== RESULTS =====")
            print(f"Bessel Fit Time: {bessel_time:.6f} seconds")
            print(f"Centroid: X={cx:.4f}, Y={cy:.4f}")

        csv_file = f"bessel-results-{path}.csv".replace("/", "").replace("\\", "")  # For linux & windows

        header = ["Image", "Bessel Time"]

        with open(csv_file, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_file}")


if __name__ == '__main__':
    dataset_directory = "images"
    image_directory_paths = []
    # Generate a list with all the directories in the root path
    # Only directories allowed in a root path is a multy thread requirement for simplicity
    for image_directory in os.listdir(dataset_directory):
        image_directory_paths.append(os.path.join(dataset_directory, image_directory))

    actual_path = []
    for actual_path in image_directory_paths:
        # process_local_dataset(actual_path)
        BesselFitter.process_local_dataset(actual_path)
