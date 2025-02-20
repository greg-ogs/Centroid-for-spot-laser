'''
Created by greg-ogs
'''
import csv
import os
import time
from threading import Thread

from centroid_calculator import superpixels, calculate_centroid, calculate_centroid_scikit

"""
This is a previous conclusion, for PID the superpixel bring a better result to detect the center of the spot in the border of the image.
For error measurement the cv or the scikit are better
A ideal algorith implements morphological operations when the spot is in reaching the center of the image
"""

def test_algorithms_on_all_images(directory_path, n_segments=100, compactness=10):
    """
    For each image in the specified directory, this test measures time taken
    by different algorithms and saves the results into a CSV file.
    """
    results = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            print(f"\n--- Testing on image: {filename} ---")

            # Initialize superpixels object for testing
            sp_centroid = superpixels(image_path, n_segments, compactness)

            # SLIC
            start = time.time()
            sp_centroid.calculate_superpixels_slic()
            elapsed_slic = time.time() - start

            # Quickshift
            start = time.time()
            sp_centroid.calculate_superpixels_quickshift()
            elapsed_quickshift = time.time() - start

            # Felzenszwalb
            start = time.time()
            sp_centroid.calculate_superpixels_felzenszwalb()
            elapsed_felzenszwalb = time.time() - start

            # OpenCV Centroid
            start = time.time()
            calculate_centroid(image_path)
            elapsed_cv2 = time.time() - start

            # scikit-image Centroid
            start = time.time()
            calculate_centroid_scikit(image_path)
            elapsed_scikit = time.time() - start

            # Append results for CSV
            results.append([
                filename, elapsed_felzenszwalb, elapsed_slic,
                elapsed_quickshift, elapsed_cv2, elapsed_scikit
            ])

            # Print the results for each image
            print(f"  SLIC time:             {elapsed_slic:.5f} s")
            print(f"  Quickshift time:       {elapsed_quickshift:.5f} s")
            print(f"  Felzenszwalb time:     {elapsed_felzenszwalb:.5f} s")
            print(f"  OpenCV centroid time:  {elapsed_cv2:.5f} s")
            print(f"  Scikit centroid time:  {elapsed_scikit:.5f} s")

    # Save results to CSV
    csv_file = "algorithm_results.csv"
    header = ["Image", "Felzenszwalb Time", "SLIC Time", "Quickshift Time", "OpenCV Centroid Time",
              "Scikit-image Centroid Time"]
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)
    print(f"\nResults saved to {csv_file}")


if __name__ == "__main__":
    dataset_directory = "images"
    image_directory_paths = []
    for image_directory in os.listdir(dataset_directory):
        image_directory_paths.append(os.path.join(dataset_directory, image_directory))
    threads = []  # List to store threads
    for path in image_directory_paths:
        thread = Thread(target=test_algorithms_on_all_images, args=(path,))
        threads.append(thread)
        thread.start()  # Start the thread

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
