import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float


class superpixels:
    """
    Represents an image processing utility for superpixels calculation using the SLIC algorithm.
    This class can process an input image and apply segmentation to generate superpixels. It also
    computes the characteristics of segments such as their central coordinates and maximum values.

    :ivar n_segments: Number of segments to generate in the superpixel calculation.
    :type n_segments: int
    :ivar compactness: Compactness parameter influencing the superpixel shapes.
    :type compactness: float
    :ivar image_ref: Path to the image to be processed.
    :type image_ref: str
    """
    def __init__(self, image_path, num_of_segments=100, a_compactness=10):
        """
        Initializes the segmentation object with image path, number of segments,
        and compactness value. These parameters are used to configure the
        Superpixel segmentation process for the input image.

        :param image_path: Specifies the path to the input image on which
            segmentation will be performed.
        :type image_path: str
        :param num_of_segments: The desired number of segments to divide the
            image into for segmentation. Default is 100.
        :type num_of_segments: int, optional
        :param a_compactness: Controls the compactness of superpixels. Higher
            values result in more square-like segments. Default is 10.
        :type a_compactness: int, optional
        """
        self.n_segments = num_of_segments
        self.compactness = a_compactness
        self.image_ref = image_path

    def calculate_superpixels_slic(self):
        """
        Calculates superpixels for the given image and displays the segmented regions using
        SLIC (Simple Linear Iterative Clustering). The function processes the image, converts
        it to grayscale if necessary, and applies the SLIC algorithm to generate superpixels.
        It also displays the segmented image with boundaries using matplotlib and calculates
        the centers of the regions.
        
        :raises ValueError: If the input image is not in a valid format.
        :return:
            None
        """

        image_data = img_as_float(imread(self.image_ref))
        image_data = np.array(image_data)
        if len(image_data.shape) == 3:
            image = rgb2gray(image_data)
        else:
            a_array = image_data
            b_array = image_data
            c_array = np.dstack((a_array, b_array))
            image = np.dstack((c_array, b_array))

        segments = slic(image, n_segments=self.n_segments, compactness=self.compactness, sigma=5)
        X, Y = self.center_of_spot(image, segments)
        print('SLIC centroid coordinates are in X = ' + str(X) + ' & Y = ' + str(Y))

        # Show the output of SLIC
        fig = plt.figure("Superpixels -- SLIC (%d segments)" % (self.n_segments))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image, segments))
        plt.plot(X, Y, marker='o', markersize=5, color='red')
        plt.title("Superpixels -- SLIC (%d segments)" % (self.n_segments))
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.axis("on")
        
        plt.show()


    def calculate_superpixels_quickshift(self):
        """
        Calculates superpixels for the given image and displays the segmented regions using
        the Quickshift segmentation technique. This method applies the Quickshift algorithm
        to generate superpixels and displays the segmented image with boundaries.

        :raises ValueError: If the input image is not in a valid format.
        :return:
            None
        """
        from skimage.segmentation import quickshift

        image_data = img_as_float(imread(self.image_ref))
        image_data = np.array(image_data)
        if len(image_data.shape) == 3:
            image = rgb2gray(image_data)
        else:
            a_array = image_data
            b_array = image_data
            c_array = np.dstack((a_array, b_array))
            image = np.dstack((c_array, b_array))

        segments = quickshift(image, kernel_size=5, max_dist=19, ratio=5)
        X, Y = self.center_of_spot(image, segments)
        print('Quick-shift centroid coordinates are in X = ' + str(X) + ' & Y = ' + str(Y) )
        # Show the output of Quickshift
        fig = plt.figure("Superpixels -- Quickshift")
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image_data, segments))
        plt.plot(X, Y, marker='o', markersize=5, color='red')
        plt.title("Superpixels -- Quickshift")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.axis("on")

        plt.show()

    def calculate_superpixels_felzenszwalb(self):
        """
        Calculates superpixels for the given image and displays the segmented regions using
        the Felzenszwalb segmentation technique. This method applies the Felzenszwalb algorithm
        to generate superpixels and displays the segmented image with boundaries.

        :raises ValueError: If the input image is not in a valid format.
        :return:
            None
        """
        from skimage.segmentation import felzenszwalb

        image_data = img_as_float(imread(self.image_ref))
        image_data = np.array(image_data)
        if len(image_data.shape) == 3:
            image = rgb2gray(image_data)
        else:
            a_array = image_data
            b_array = image_data
            c_array = np.dstack((a_array, b_array))
            image = np.dstack((c_array, b_array))

        segments = felzenszwalb(image, scale=300, sigma=0.5, min_size=200)
        X, Y = self.center_of_spot(image, segments)
        print('Felzenszwalb centroid coordinates are in X = ' + str(X) + ' & Y = ' + str(Y) )
        # Show the output of Felzenszwalb
        fig = plt.figure("Superpixels -- Felzenszwalb")
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(image_data, segments))
        plt.plot(X, Y, marker='o', markersize=5, color='red')
        plt.title("Superpixels -- Felzenszwalb")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.axis("on")

        plt.show()

    @staticmethod
    def center_of_spot(image, segments):
        """
        Computes the center coordinates of a spot within a segmented image by identifying the
        segment with the highest mean value of pixel intensities and locating its approximate
        center position.

        :param image: A 3D numpy array representing the image, where pixel intensity values
                      are represented in the first channel.
        :type image: numpy.ndarray
        :param segments: A 2D numpy array representing the segmentation map, where each
                         segment is identified by an integer label.
        :type segments: numpy.ndarray

        :return: A tuple containing the x and y coordinates of the center of the segment
                 with the maximum mean value. These coordinates approximate the center
                 position of the segment.
        :rtype: tuple
        """
        # mean segments
        # lower/ more pre
        nlabels = np.amax(segments)
        nlabels = nlabels + 1
        nlabels = int(nlabels)
        values = []
        for i in range(1, nlabels):
            coor = np.where(segments == i)  # coordenada de cada segmento
            # co = [coor[0][0],coor[1][0]]# toma la primera coordenada de cada segmento
            # segmentVal = image[co[0]][co[1]][2]#usa la coordenada anterior para buscar el valor en la imagen
            arraysize = coor[0].shape  # canntidad de coordenadas de el segmento actual
            arrsiz = arraysize[0]
            meansum = []
            for j in range(arrsiz):
                # individualCoor = [coor[0][j],coor[1][j]]#coordenada individual de cda pixel del segemento
                coorVal = image[coor[0][j]][coor[1][j]][0]  # valaor de cada pixel del segmento
                meansum.append(coorVal)  # se agrega el valor a un vector
            segmentVal = np.mean(meansum)  # media
            values.append(segmentVal)  # agrega ese valor a una variable (la media de cada segmento)
        maxsegment = np.where(values == np.amax(values))  # elige segmento con valor maximo
        maxS = maxsegment[0] + 1  # compensacion del 0 en el indice del array
        maxseg = maxS[0]
        # print(maxseg)
        maxVC = np.where(
            segments == maxseg)  # selecciona todas las coordenadas del segmento con valor maximo
        # calcular la distancia desde el segmento hasta el centro
        arraysz = maxVC[0].shape  # dimencion del conjunto de coordenadas del segmento
        arsz = int(arraysz[0] / 2)  # la mitad de ese conjunto
        XselectCoor = maxVC[1][arsz]  # coordenada intermedia en x
        X = XselectCoor
        YselectCoor = maxVC[0][arsz]  # coordenada intermedia en y
        Y = YselectCoor
        return X, Y

def calculate_centroid(image_path):
    """
    Calculates the centroid of the largest object in the provided image.

    This function processes the input image to detect and isolate the largest object.
    It utilizes computer vision techniques such as grayscale conversion, Gaussian
    blurring, thresholding, and morphological operations to clean the image and
    enhance object detection. Once the largest object is identified via contours,
    the centroid is computed using image moments.

    :param image_path: Path to the image file.
    :type image_path: str
    :return: A tuple containing the x and y coordinates of the centroid of the
        largest object, or None if no objects are detected.
    :rtype: tuple[int, int] | None
    """
    # Step 1: Read the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply Gaussian Blur to reduce noise (optional)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 4: Threshold the image to create a binary version
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # Step 5: Apply morphological operations to clean and isolate the object
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Step 6: Find contours in the binary image
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure there's at least one contour detected
    if len(contours) == 0:
        print("No objects detected!")
        return None

    # Step 7: Assume the largest contour is the object
    largest_contour = max(contours, key=cv2.contourArea)

    # Step 8: Calculate moments to find the centroid
    moments = cv2.moments(largest_contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])  # X-coordinate of the centroid
        cy = int(moments['m01'] / moments['m00'])  # Y-coordinate of the centroid
    else:
        # In case of a single point as an object (unlikely in this case)
        cx, cy = 0, 0

    print(f"Centroid of the object is at: ({cx}, {cy})")

    # Step 9: Optional - Display the result
    result_image = image.copy()
    cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 2)  # Draw the largest contour
    cv2.circle(result_image, (cx, cy), 5, (0, 0, 255), -1)  # Mark the centroid with a red circle
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
    plt.title("Centroid calculated with FBM (OpenCV)")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.axis("on")
    plt.show()

    return cx, cy

def calculate_centroid_scikit(image_path):
    """
    Calculates the centroid of the largest connected region in a given image using scikit-image.

    This function performs preprocessing on the image, including Gaussian Blur and morphological
    operations, and then identifies the largest connected region. The centroid of this region
    is computed and displayed. If no connected regions are found, it returns None.

    :param str image_path: The path to the image file to be processed.
    :return: A tuple containing the x and y coordinates of the centroid as integers.
    :rtype: tuple[int, int] | None
    """
    # Read the image using scikit-image
    image = imread(image_path)

    if len(image.shape) == 2:
        gray = image
    else:
        gray = image[:, :, 0]

    # Apply Gaussian Blur
    blurred = gaussian(gray, sigma=2)  # Adjust sigma as needed

    # Threshold the image
    thresh = threshold_otsu(blurred)
    binary = blurred > thresh

    # Morphological closing
    selem = disk(5)  # Adjust disk size as needed
    morph = closing(binary, selem)

    # Label connected regions
    label_image = label(morph)

    # Find the largest contour
    regions = regionprops(label_image)
    if not regions:
        print("No objects detected!")
        return None

    largest_region = max(regions, key=lambda r: r.area)

    # Calculate centroid
    cy, cx = largest_region.centroid  # Note: skimage returns (row, col)

    print(f"Centroid of the object is at: ({int(cx)}, {int(cy)})")

    # Display the result (optional)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.plot(cx, cy, 'o', markersize=5, color='red')  # Mark the centroid with a red circle
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.axis("on")
    plt.title("Centrid calculated with CCL (Scikit-image)")
    plt.show()
    return int(cx), int(cy)

if __name__ == '__main__':
    image_path = "images/dw/image100.png"
    superpixels_centroid = superpixels(image_path, 100, 10)
    # Superpixels
    # felzenszwalb

    start = time.time()
    superpixels_centroid.calculate_superpixels_slic()
    end = time.time()
    superpixels_felzenszwalb_time = end - start

    # SLIC

    start = time.time()
    superpixels_centroid.calculate_superpixels_slic()
    end = time.time()
    superpixels_SLIC_time = end - start

    # quickshift

    reset = time.time()
    superpixels_centroid.calculate_superpixels_quickshift()
    end = time.time()
    superpixels_quickshift_time = end - reset

    # cv2 centroid

    reset = time.time()
    calculate_centroid(image_path)
    end = time.time()
    cv2_centroid_time = end - reset

    # Scikit centroid

    reset = time.time()
    calculate_centroid_scikit(image_path)
    end = time.time()
    scikit_centroid_time = end - reset

    print("Superpixels slic time: " + str(superpixels_SLIC_time))
    print("Superpixels felzenszwalb time: " + str(superpixels_felzenszwalb_time))
    print("Superpixels quickshift time: " + str(superpixels_quickshift_time))
    print("cv2 centroid time: " + str(cv2_centroid_time))
    print("scikit centroid time: " + str(scikit_centroid_time))
