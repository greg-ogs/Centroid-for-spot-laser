from centroid_calculator import *
import time

if __name__ == '__main__':
    image_path = "images/image100.png"
    superpixels_centroid = superpixels(image_path, 100, 10)
    # Superpixels
    #felzenszwalb

    # start = time.time()
    # superpixels_centroid.calculate_superpixels_slic()
    # end = time.time()
    # superpixels_felzenszwalb_time = end - start

    # SLIC

    # start = time.time()
    # superpixels_centroid.calculate_superpixels_slic()
    # end = time.time()
    # superpixels_SLIC_time = end - start

    #quickshift

    reset = time.time()
    superpixels_centroid.calculate_superpixels_quickshift()
    end = time.time()
    superpixels_quickshift_time = end - reset

    #cv2 centroid

    # reset = time.time()
    # calculate_centroid(image_path)
    # end = time.time()
    # cv2_centroid_time = end - reset

    #Scikit centroid

    # reset = time.time()
    # calculate_centroid_scikit(image_path)
    # end = time.time()
    # scikit_centroid_time = end - reset

    # print("Superpixels slic time: " + str(superpixels_SLIC_time))
    # print("Superpixels felzenszwalb time: " + str(superpixels_felzenszwalb_time))
    print("Superpixels quickshift time: " + str(superpixels_quickshift_time))
    # print("cv2 centroid time: " + str(cv2_centroid_time))
    # print("scikit centroid time: " + str(scikit_centroid_time))
