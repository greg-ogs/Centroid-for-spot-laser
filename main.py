"""
This main.py script is used to process all images in the images directory and calculate the average processing times for
each algorithm.
Also creates a CSV file with the results and plots for each image.
Each image plot is only for visualization purposes, must be commented for actual comparison tests.
The script uses the centroid_calculator.py file to calculate the centroid for each image.
"""

import csv
from centroid_calculator import *
import os
from threading import Thread

def process_local_dataset(path):
    image_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".png"):
                # Skip result images (those that are named after algorithms)
                if not any(file.startswith(alg_result) for alg_result in ["CCL", "FBM", "Felzenszwalb", "Quickshift", "SLIC"]):
                    # Use forward slashes for compatibility with the existing code
                    file_path = os.path.join(root, file).replace("\\", "/")
                    image_files.append(file_path)

    print(f"Found {len(image_files)} images to process:")
    for img in image_files:
        print(f"- {img}")

    # Process each image
    results = []
    for image_path in image_files:
        print(f"\nProcessing image: {image_path}")
        superpixels_centroid = Superpixels(image_path, 100, 10)

        # SLIC
        reset = time.time()
        superpixels_centroid.calculate_superpixels_slic()
        end = time.time()
        superpixels_slic_time = end - reset

        # Superpixels - felzenszwalb
        reset = time.time()
        superpixels_centroid.calculate_superpixels_felzenszwalb()
        end = time.time()
        superpixels_felzenszwalb_time = end - reset

        # Quickshift
        reset = time.time()
        superpixels_centroid.calculate_superpixels_quickshift()
        end = time.time()
        superpixels_quickshift_time = end - reset

        # FBM
        reset = time.time()
        calculate_centroid(image_path)
        end = time.time()
        fbm_time = end - reset

        # CCL
        reset = time.time()
        calculate_centroid_scikit(image_path)
        end = time.time()
        ccl_time = end - reset

        # Print results for this image
        print(f"Results for {image_path}:")
        print("Superpixels slic time: " + str(superpixels_slic_time))
        print("Superpixels felzenszwalb time: " + str(superpixels_felzenszwalb_time))
        print("Superpixels quickshift time: " + str(superpixels_quickshift_time))
        print("FBM time: " + str(fbm_time))
        print("CCL time: " + str(ccl_time))

        result = {
            'Image': image_path,
            'SLIC Time': superpixels_slic_time,
            'Felzenszwalb Time': superpixels_felzenszwalb_time,
            'Quickshift Time': superpixels_quickshift_time,
            'FBM Time': fbm_time,
            'CCL Time': ccl_time
        }
        results.append(result)

    csv_file = f"results-{path}.csv".replace("/", "").replace("\\", "") # For linux & windows

    header = ["Image", "SLIC Time", "Felzenszwalb Time", "Quickshift Time", "FBM Time", "CCL Time"]

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_file}")

    # Print summary
    print("\n===== SUMMARY =====")
    print(f"Processed {len(results)} images")

    # Calculate averages
    if results:
        avg_slic = sum(r['SLIC Time'] for r in results) / len(results)
        avg_felzenszwalb = sum(r['Felzenszwalb Time'] for r in results) / len(results)
        avg_quickshift = sum(r['Quickshift Time'] for r in results) / len(results)
        avg_fbm = sum(r['FBM Time'] for r in results) / len(results)
        avg_ccl = sum(r['CCL Time'] for r in results) / len(results)

        print("\nAverage processing times:")
        print(f"Superpixels SLIC: {avg_slic:.6f} seconds")
        print(f"Superpixels Felzenszwalb: {avg_felzenszwalb:.6f} seconds")
        print(f"Superpixels Quickshift: {avg_quickshift:.6f} seconds")
        print(f"FBM: {avg_fbm:.6f} seconds")
        print(f"Scikit centroid: {avg_ccl:.6f} seconds")


if __name__ == '__main__':
    # Find all PNG images in the images directory and its subdirectories
    dataset_directory = "images"
    image_directory_paths = []
    # Generate a list with all the directories in the root path
    # Only directories allowed in a root path is a multy thread requirement for simplicity
    for image_directory in os.listdir(dataset_directory):
        image_directory_paths.append(os.path.join(dataset_directory, image_directory))

    actual_path = []
    threads = []  # List to store threads
    for actual_path in image_directory_paths:
        process_local_dataset(actual_path)
        # thread = Thread(target=process_local_dataset, args=(actual_path,))
        # threads.append(thread)
        # thread.start()

