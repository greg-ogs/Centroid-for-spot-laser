from centroid_calculator import *
import os

def process_image(image_path):
    print(f"\nProcessing image: {image_path}")
    superpixels_centroid = superpixels(image_path, 100, 10)

    # Superpixels - felzenszwalb
    start = time.time()
    superpixels_centroid.calculate_superpixels_felzenszwalb()
    end = time.time()
    superpixels_felzenszwalb_time = end - start

    # SLIC
    start = time.time()
    superpixels_centroid.calculate_superpixels_slic()
    end = time.time()
    superpixels_SLIC_time = end - start

    # Quickshift
    reset = time.time()
    superpixels_centroid.calculate_superpixels_quickshift()
    end = time.time()
    superpixels_quickshift_time = end - reset

    # FBM
    reset = time.time()
    calculate_centroid(image_path)
    end = time.time()
    FBM_time = end - reset

    # CCL
    reset = time.time()
    calculate_centroid_scikit(image_path)
    end = time.time()
    CCL_time = end - reset

    # Print results for this image
    print(f"Results for {image_path}:")
    print("Superpixels slic time: " + str(superpixels_SLIC_time))
    print("Superpixels felzenszwalb time: " + str(superpixels_felzenszwalb_time))
    print("Superpixels quickshift time: " + str(superpixels_quickshift_time))
    print("FBM time: " + str(FBM_time))
    print("CCL time: " + str(CCL_time))

    return {
        'image': image_path,
        'slic': superpixels_SLIC_time,
        'felzenszwalb': superpixels_felzenszwalb_time,
        'quickshift': superpixels_quickshift_time,
        'FBM': FBM_time,
        'CCL': CCL_time
    }

if __name__ == '__main__':
    # Find all PNG images in the images directory and its subdirectories
    image_files = []
    for root, dirs, files in os.walk("images"):
        for file in files:
            if file.endswith(".png"):
                # Skip result images (those that are named after algorithms)
                if not any(file.startswith(alg_result) for alg_result in ["CCL", "FBM", "Felzenszwalb", "Quickshift", "SLIC"]):
                    # Use forward slashes for compatibility with the existing code
                    image_path = os.path.join(root, file).replace("\\", "/")
                    image_files.append(image_path)

    print(f"Found {len(image_files)} images to process:")
    for img in image_files:
        print(f"- {img}")

    # Process each image
    results = []
    for image_path in image_files:
        result = process_image(image_path)
        results.append(result)

    # Print summary
    print("\n===== SUMMARY =====")
    print(f"Processed {len(results)} images")

    # Calculate averages
    if results:
        avg_slic = sum(r['slic'] for r in results) / len(results)
        avg_felzenszwalb = sum(r['felzenszwalb'] for r in results) / len(results)
        avg_quickshift = sum(r['quickshift'] for r in results) / len(results)
        avg_FBM = sum(r['FBM'] for r in results) / len(results)
        avg_CCL = sum(r['CCL'] for r in results) / len(results)

        print("\nAverage processing times:")
        print(f"Superpixels SLIC: {avg_slic:.6f} seconds")
        print(f"Superpixels Felzenszwalb: {avg_felzenszwalb:.6f} seconds")
        print(f"Superpixels Quickshift: {avg_quickshift:.6f} seconds")
        print(f"FBM: {avg_FBM:.6f} seconds")
        print(f"Scikit centroid: {avg_CCL:.6f} seconds")
