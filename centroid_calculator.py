import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import closing, disk
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float

class Superpixels:
    """
    Represents an image processing utility for Superpixels calculation using the SLIC algorithm.
    This class can process an input image and apply segmentation to generate Superpixels. It also
    computes the characteristics of segments such as their central coordinates and maximum values.

    :ivar n_segments: Number of segments to generate in the superpixel calculation.
    :type n_segments: int
    :ivar compactness: Compactness parameter influencing the superpixel shapes.
    :type compactness: float
    :ivar image_ref: Path to the image to be processed.
    :type image_ref: str
    """
    def __init__(self, image_path, num_of_segments=75, a_compactness=10):
        """
        Initializes the segmentation object with image path, number of segments,
        and compactness value. These parameters are used to configure the
        Superpixel segmentation process for the input image.

        :param image_path: Specifies the path to the input image on which
            segmentation will be performed.
        :type image_path: str
        :param num_of_segments: The desired number of segments to divide the
            image into for segmentation. Default is 100.
        :type num_of_segments: int, optionally
        :param a_compactness: Controls the compactness of Superpixels. Higher
            values result in more square-like segments. Default is 10.
        :type a_compactness: int, optional
        """
        self.n_segments = num_of_segments
        self.compactness = a_compactness
        self.image_ref = image_path

        image_data = img_as_float(imread(image_path))

        # Ensure we have a 2D grayscale image for the 3D plot's Z-axis
        if image_data.ndim == 2:
            gray_image = image_data
        else:
            if image_data.shape[2] == 4:  # Handle RGBA images
                image_data = image_data[:, :, :3]
            gray_image = rgb2gray(image_data)

        # For superpixels and center_of_spot, use a 3-channel version of the grayscale image
        image_for_super_process = np.dstack([gray_image] * 3)

        self.superpixels_images = [gray_image, image_for_super_process]

    @staticmethod
    def plot_wireframe(actual_algorithm, rotated_gray_image_meth, x_grid_meth, y_grid_meth, rotated_segments_meth, h_rot_meth, w_rot_meth, x_meth, y_meth, stride=10):
        # Plot the center using a wireframe

        # Create 3D plot
        fig_3d = plt.figure(f"3D Visualization -- {actual_algorithm}", figsize=(15, 15))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # Plot the 3D surface
        # Use a smaller stride for more detail
        # Scale the z-axis (intensity) to 256
        scaled_intensity = rotated_gray_image_meth[::stride, ::stride] * 256
        ax_3d.plot_wireframe(x_grid_meth[::stride, ::stride], y_grid_meth[::stride, ::stride],
                           scaled_intensity,
                           cmap='viridis', alpha=0.7, linewidth=0)

        # Plot superpixel boundaries on the surface
        for label in np.unique(rotated_segments_meth):
            contours = find_contours(rotated_segments_meth, level=label)
            for contour in contours:
                y_contour, x_contour = contour[:, 0].astype(int), contour[:, 1].astype(int)
                y_contour = np.clip(y_contour, 0, h_rot_meth - 1)
                x_contour = np.clip(x_contour, 0, w_rot_meth - 1)
                # Scale the z-axis (intensity) to 256 and raise slightly for visibility
                z_contour = rotated_gray_image_meth[y_contour, x_contour] * 256 + 2.5  # Raise slightly
                ax_3d.plot(x_contour, y_contour, z_contour, color='black', linewidth=1.5)

        ax_3d.scatter(y_meth, 1280 - x_meth, 256, c='red', s=250, marker='o', depthshade=True,
                      label='Centroid')

        # ax_3d.set_xlabel('X')
        # ax_3d.set_ylabel('Y')
        # ax_3d.set_zlabel('Intensity')
        ax_3d.view_init(elev=50, azim=280)
        ax_3d.legend(fontsize=20)
        plt.savefig(f'{actual_algorithm}wireframe.png')
        plt.show()

    def calculate_superpixels_slic(self):
        """
        Calculates Superpixels for the given image and displays the segmented regions using
        SLIC (Simple Linear Iterative Clustering). The function processes the image, converts
        it to grayscale if necessary, and applies the SLIC algorithm to generate Superpixels.
        It also displays the segmented image with boundaries using matplotlib and calculates
        the centers of the regions.
        
        :raises ValueError: If the input image is not in a valid format.
        :return:
            None
        """

        gray_image_2d, image = self.superpixels_images

        segments = slic(image, n_segments=self.n_segments, compactness=self.compactness, sigma=5)
        x, y = self.center_of_spot(image, segments)
        print('SLIC centroid coordinates are in X = ' + str(x) + ' & Y = ' + str(y))

        # Show the output of SLIC
        plt.rcParams.update({'font.size': 30})
        fig = plt.figure("Superpixels -- SLIC (%d segments)" % self.n_segments, figsize=(11, 12.8))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(np.rot90(mark_boundaries(image, segments)), origin='lower')
        plt.plot(y, 1280-x, marker='o', markersize=15, color='red')  # Swap X and Y for the rotated plot
        # plt.title("Superpixels -- SLIC (%d segments)" % (self.n_segments))
        plt.xlabel("pixels")
        plt.ylabel("pixels")
        plt.axis("on")
        plt.savefig('SLIC.png')
        plt.show()

        # Show the output of slic as 3d graphic

        # h, w = gray_image_2d.shape

        # Rotate image and segments for consistency with 2D plot
        rotated_gray_image = np.rot90(gray_image_2d)
        rotated_segments = np.rot90(segments)
        h_rot, w_rot = rotated_gray_image.shape
        y_grid, x_grid = np.mgrid[0:h_rot, 0:w_rot]

        # Create 3D plot
        fig_3d = plt.figure("3D Visualization -- SLIC", figsize=(15, 15))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # Plot the 3D surface
        stride = 1  # Use a smaller stride for more detail
        # Scale the z-axis (intensity) to 256
        scaled_intensity = rotated_gray_image[::stride, ::stride] * 256
        ax_3d.plot_surface(x_grid[::stride, ::stride], y_grid[::stride, ::stride],
                           scaled_intensity,
                           cmap='viridis', alpha=0.7, linewidth=0)

        # Plot superpixel boundaries on the surface
        for label in np.unique(rotated_segments):
            contours = find_contours(rotated_segments, level=label)
            for contour in contours:
                y_contour, x_contour = contour[:, 0].astype(int), contour[:, 1].astype(int)
                y_contour = np.clip(y_contour, 0, h_rot - 1)
                x_contour = np.clip(x_contour, 0, w_rot - 1)
                # Scale the z-axis (intensity) to 256 and raise slightly for visibility
                z_contour = rotated_gray_image[y_contour, x_contour] * 256 + 2.5  # Raise slightly
                ax_3d.plot(x_contour, y_contour, z_contour, color='black', linewidth=1.5)
        plt.savefig(f'SLIC-surface.png')
        plt.show()

        # Plot the center using a wireframe
        self.plot_wireframe("SLIC", rotated_gray_image, x_grid, y_grid, rotated_segments,
                            h_rot, w_rot, x, y)


    def calculate_superpixels_quickshift(self):
        """
        Calculates Superpixels for the given image and displays the segmented regions using
        the Quickshift segmentation technique. This method applies the Quickshift algorithm
        to generate Superpixels and displays the segmented image with boundaries.

        :raises ValueError: If the input image is not in a valid format.
        :return:
            None
        """
        from skimage.segmentation import quickshift

        gray_image_2d, image = self.superpixels_images

        segments = quickshift(image, kernel_size=21, max_dist=50, ratio=5)
        x, y = self.center_of_spot(image, segments)
        print('Quick-shift centroid coordinates are in X = ' + str(x) + ' & Y = ' + str(y) )
        # Show the output of Quickshift
        plt.rcParams.update({'font.size': 30})
        plt.rcParams['figure.figsize'] = 11, 12.8
        fig = plt.figure("Superpixels -- Quickshift")
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(np.rot90(mark_boundaries(image, segments)), origin='lower')
        plt.plot(y, 1280-x, marker='o', markersize=15, color='red')
        # plt.title("Superpixels -- Quickshift")
        plt.xlabel("pixels")
        plt.ylabel("pixels")
        plt.axis("on")
        plt.savefig('Quickshift.png')
        plt.show()

        # Show the output of quickshift as a 3d graphic

        # h, w = gray_image_2d.shape

        # Rotate image and segments for consistency with 2D plot
        rotated_gray_image = np.rot90(gray_image_2d)
        rotated_segments = np.rot90(segments)
        h_rot, w_rot = rotated_gray_image.shape
        y_grid, x_grid = np.mgrid[0:h_rot, 0:w_rot]

        # Create 3D plot
        fig_3d = plt.figure("3D Visualization -- SLIC", figsize=(15, 15))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # Plot the 3D surface
        stride = 1  # Use a smaller stride for more detail
        # Scale the z-axis (intensity) to 256
        scaled_intensity = rotated_gray_image[::stride, ::stride] * 256
        ax_3d.plot_surface(x_grid[::stride, ::stride], y_grid[::stride, ::stride],
                           scaled_intensity,
                           cmap='viridis', alpha=0.7, linewidth=0)

        # Plot superpixel boundaries on the surface
        for label in np.unique(rotated_segments):
            contours = find_contours(rotated_segments, level=label)
            for contour in contours:
                y_contour, x_contour = contour[:, 0].astype(int), contour[:, 1].astype(int)
                y_contour = np.clip(y_contour, 0, h_rot - 1)
                x_contour = np.clip(x_contour, 0, w_rot - 1)
                # Scale the z-axis (intensity) to 256 and raise slightly for visibility
                z_contour = rotated_gray_image[y_contour, x_contour] * 256 + 2.5  # Raise slightly
                ax_3d.plot(x_contour, y_contour, z_contour, color='black', linewidth=1.5)
        plt.savefig(f'quick-surface.png')
        plt.show()

        # Plot the center using a wireframe
        self.plot_wireframe("Quickshift", rotated_gray_image, x_grid, y_grid, rotated_segments,
                            h_rot, w_rot, x, y)

    def calculate_superpixels_felzenszwalb(self):
        """
        Calculates Superpixels for the given image and displays the segmented regions using
        the Felzenszwalb segmentation technique. This method applies the Felzenszwalb algorithm
        to generate Superpixels and displays the segmented image with boundaries.

        :raises ValueError: If the input image is not in a valid format.
        :return:
            None
        """
        from skimage.segmentation import felzenszwalb

        gray_image_2d, image = self.superpixels_images

        segments = felzenszwalb(image, scale=300, sigma=0.5, min_size=200)
        x, y = self.center_of_spot(image, segments)
        print('Felzenszwalb centroid coordinates are in X = ' + str(x) + ' & Y = ' + str(y) )
        # Show the output of Felzenszwalb
        plt.rcParams.update({'font.size': 30})
        plt.rcParams['figure.figsize'] = 11, 12.8
        fig = plt.figure("Superpixels -- Felzenszwalb")
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(np.rot90(mark_boundaries(image, segments)), origin='lower')
        plt.plot(y, 1280-x, marker='o', markersize=15, color='red')
        # plt.title("Superpixels -- Felzenszwalb")
        plt.xlabel("pixels")
        plt.ylabel("pixels")
        plt.axis("on")
        plt.savefig('Felzenszwalb.png')
        plt.show()

        # Show the output of felzenszwalb as 3d graphic

        # h, w = gray_image_2d.shape

        # Rotate image and segments for consistency with 2D plot
        rotated_gray_image = np.rot90(gray_image_2d)
        rotated_segments = np.rot90(segments)
        h_rot, w_rot = rotated_gray_image.shape
        y_grid, x_grid = np.mgrid[0:h_rot, 0:w_rot]

        # Create 3D plot
        fig_3d = plt.figure("3D Visualization -- SLIC", figsize=(15, 15))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # Plot the 3D surface
        stride = 1  # Use a smaller stride for more detail
        # Scale the z-axis (intensity) to 256
        scaled_intensity = rotated_gray_image[::stride, ::stride] * 256
        ax_3d.plot_surface(x_grid[::stride, ::stride], y_grid[::stride, ::stride],
                           scaled_intensity,
                           cmap='viridis', alpha=0.7, linewidth=0)

        # Plot superpixel boundaries on the surface
        for label in np.unique(rotated_segments):
            contours = find_contours(rotated_segments, level=label)
            for contour in contours:
                y_contour, x_contour = contour[:, 0].astype(int), contour[:, 1].astype(int)
                y_contour = np.clip(y_contour, 0, h_rot - 1)
                x_contour = np.clip(x_contour, 0, w_rot - 1)
                # Scale the z-axis (intensity) to 256 and raise slightly for visibility
                z_contour = rotated_gray_image[y_contour, x_contour] * 256 + 2.5  # Raise slightly
                ax_3d.plot(x_contour, y_contour, z_contour, color='black', linewidth=1.5)

        plt.savefig(f'Felzenszwalb-surface.png')
        plt.show()
        # Plot the center using a wireframe
        self.plot_wireframe("Felzenszwalb", rotated_gray_image, x_grid, y_grid, rotated_segments,
                            h_rot, w_rot, x, y)

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
                coor_val = image[coor[0][j]][coor[1][j]][0]  # valaor de cada pixel del segmento
                meansum.append(coor_val)  # se agrega el valor a un vector
            segment_val = np.mean(meansum)  # promedio de valores para cada segmento
            values.append(segment_val)  # agrega ese valor a una variable (la media de cada segmento)
        maxsegment = np.where(values == np.amax(values))  # elige segmento con valor maximo
        max_s = maxsegment[0] + 1  # compensacion del 0 en el indice del array
        maxseg = max_s[0]
        # print(maxseg)
        max_vc = np.where(
            segments == maxseg)  # selecciona todas las coordenadas del segmento con valor maximo
        # calcular la distancia desde el segmento hasta el centro
        arraysz = max_vc[0].shape  # dimencion del conjunto de coordenadas del segmento
        arsz = int(arraysz[0] / 2)  # la mitad de ese conjunto
        x_select_coor = max_vc[1][arsz]  # coordenada intermedia en x
        x = x_select_coor
        y_select_coor = max_vc[0][arsz]  # coordenada intermedia en y
        y = y_select_coor
        return x, y

def calculate_centroid(fbm_image_path):
    """
    Calculates the centroid of the largest object in the provided image.

    This function processes the input image to detect and isolate the largest object.
    It utilizes computer vision techniques such as grayscale conversion, Gaussian
    blurring, thresholding, and morphological operations to clean the image and
    enhance object detection. Once the largest object is identified via contours,
    the centroid is computed using image moments.

    :param fbm_image_path: Path to the image file.
    :type fbm_image_path: str
    :return: A tuple containing the x and y coordinates of the centroid of the
        largest object, or None if no objects are detected.
    :rtype: tuple[int, int] | None
    """
    # Step 1: Read the image
    image = cv2.imread(fbm_image_path, cv2.IMREAD_COLOR)

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
        # In the case of a single point as an object (unlikely in this case)
        cx, cy = 0, 0

    print(f"Centroid of the object is at: ({cx}, {cy})")

    # Step 9: Optional - Display the result
    result_image = image.copy()
    cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 2)  # Draw the largest contour
    cv2.circle(result_image, (cx, cy), 15, (0, 0, 255), -1)  # Mark the centroid with a red circle
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['figure.figsize'] = 11, 12.8
    plt.imshow(np.rot90(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)), origin='lower')  # Convert BGR to RGB for matplotlib
    # plt.title("Centroid calculated with FBM")
    plt.xlabel("pixeles")
    plt.ylabel("pixeles")
    plt.axis("on")
    plt.savefig('FBM.png')
    plt.show()

    # 3D visualization for FBM
    # Prepare rotated grayscale image normalized to [0,1]
    rotated_gray_image = np.rot90(gray.astype(np.float32) / 255.0)
    h_rot, w_rot = rotated_gray_image.shape
    y_grid, x_grid = np.mgrid[0:h_rot, 0:w_rot]

    # Build segmentation from morph mask for boundary plotting
    label_image = label(morph > 0)
    rotated_segments = np.rot90(label_image)

    # Plot 3D surface
    fig_3d = plt.figure("3D Visualization -- FBM", figsize=(15, 15))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    stride = 1
    scaled_intensity = rotated_gray_image[::stride, ::stride] * 256
    ax_3d.plot_surface(x_grid[::stride, ::stride], y_grid[::stride, ::stride],
                       scaled_intensity,
                       cmap='viridis', alpha=0.7, linewidth=0)
    plt.savefig('FBM-surface.png')
    plt.show()

    # Plot wireframe with centroid overlay
    Superpixels.plot_wireframe("FBM", rotated_gray_image, x_grid, y_grid, rotated_segments, h_rot, w_rot, cx, cy, 100)

    return cx, cy

def calculate_centroid_scikit(ccl_image_path):
    """
    Calculates the centroid of the largest connected region in a given image using scikit-image.

    This function performs preprocessing on the image, including Gaussian Blur and morphological
    operations, and then identifies the largest connected region. The centroid of this region
    is computed and displayed. If no connected regions are found, it returns None.

    :param str ccl_image_path: The path to the image file to be processed.
    :return: A tuple containing the x and y coordinates of the centroid as integers.
    :rtype: tuple[int, int] | None
    """
    # Read the image using scikit-image
    image = imread(ccl_image_path)

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
    plt.rcParams.update({'font.size': 30})
    plt.rcParams['figure.figsize'] = 11, 12.8
    fig, ax = plt.subplots()
    ax.imshow(np.rot90(image), origin='lower')
    ax.plot(cy, 1280-cx, 'o', markersize=15, color='red')  # Mark the centroid with a red circle
    plt.xlabel("pixeles")
    plt.ylabel("pixeles")
    plt.axis("on")
    # plt.title("Centrid calculated with CCL")
    plt.savefig('CCL.png')
    plt.show()

    # 3D visualization for CCL
    # Prepare rotated grayscale image normalized to [0,1]
    rotated_gray_image = np.rot90(img_as_float(gray))
    h_rot, w_rot = rotated_gray_image.shape
    y_grid, x_grid = np.mgrid[0:h_rot, 0:w_rot]

    # Use the label_image for boundaries
    rotated_segments = np.rot90(label_image)

    # Plot the 3D surface with boundaries overlay
    fig_3d = plt.figure("3D Visualization -- CCL", figsize=(15, 15))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    stride = 1
    scaled_intensity = rotated_gray_image[::stride, ::stride] * 256
    ax_3d.plot_surface(x_grid[::stride, ::stride], y_grid[::stride, ::stride],
                       scaled_intensity,
                       cmap='viridis', alpha=0.7, linewidth=0)

    for label_val in np.unique(rotated_segments):
        contours = find_contours(rotated_segments, level=label_val)
        for contour in contours:
            y_contour, x_contour = contour[:, 0].astype(int), contour[:, 1].astype(int)
            y_contour = np.clip(y_contour, 0, h_rot - 1)
            x_contour = np.clip(x_contour, 0, w_rot - 1)
            z_contour = rotated_gray_image[y_contour, x_contour] * 256 + 2.5
            ax_3d.plot(x_contour, y_contour, z_contour, color='black', linewidth=1.5)

    plt.savefig('CCL-surface.png')
    plt.show()

    # Plot wireframe with centroid overlay
    Superpixels.plot_wireframe("CCL", rotated_gray_image, x_grid, y_grid, rotated_segments, h_rot, w_rot, int(cx),
                               int(cy), 1)

    return int(cx), int(cy)


if __name__ == '__main__':
    path_to_image = "images/l0/image100.png"
    superpixels_centroid = Superpixels(path_to_image, 50, 10)

    # Superpixels
    # SLIC

    start = time.time()
    superpixels_centroid.calculate_superpixels_slic()
    end = time.time()
    superpixels_SLIC_time = end - start

    # Felzenszwalb

    start = time.time()
    superpixels_centroid.calculate_superpixels_felzenszwalb()
    end = time.time()
    superpixels_felzenszwalb_time = end - start

    # quickshift

    reset = time.time()
    superpixels_centroid.calculate_superpixels_quickshift()
    end = time.time()
    superpixels_quickshift_time = end - reset

    # cv2 centroid

    reset = time.time()
    calculate_centroid(path_to_image)
    end = time.time()
    cv2_centroid_time = end - reset

    # Scikit centroid

    reset = time.time()
    calculate_centroid_scikit(path_to_image)
    end = time.time()
    scikit_centroid_time = end - reset

    print("Superpixels slic time: " + str(superpixels_SLIC_time))
    print("Superpixels felzenszwalb time: " + str(superpixels_felzenszwalb_time))
    print("Superpixels quickshift time: " + str(superpixels_quickshift_time))
    print("cv2 centroid time: " + str(cv2_centroid_time))
    print("scikit centroid time: " + str(scikit_centroid_time))
