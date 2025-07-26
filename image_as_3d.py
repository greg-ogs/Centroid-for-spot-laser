'''
Created by greg-ogs at 07/16/2025
'''

import os
import skimage as ski
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def read_gray(img_path):
    img = ski.io.imread(img_path)
    img = ski.util.img_as_ubyte(img)
    if len(img.shape) > 2:
        img = ski.color.rgb2gray(img)
    # plt.figure(figsize=(10, 8))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()
    return img

def convert_to_3d(img):
    stride = 10
    # Create coordinate matrices
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    # Create 3D plot
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    # Plot wireframe
    # wire = ax.plot_wireframe(x[::stride, ::stride],
    #                          y[::stride, ::stride],
    #                          img[::stride, ::stride],
    #                          color='red')

    # Plot Surface
    wire = ax.plot_surface(x[::stride, ::stride],
                             y[::stride, ::stride],
                             img[::stride, ::stride],
                             color='#FFF000', cmap='viridis')

    ax.set_xlabel('X')
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')

    ax.view_init(elev=25, azim=280)

    plt.show()

if __name__ == "__main__":
    image_files = []
    for root, dirs, files in os.walk("images"):
        for file in files:
            if file.endswith(".png"):
                # Skip result images (those that are named after algorithms)
                if not any(file.startswith(alg_result) for alg_result in
                           ["CCL", "FBM", "Felzenszwalb", "Quickshift", "SLIC"]):
                    # Use forward slashes for compatibility with the existing code
                    image_path = os.path.join(root, file).replace("\\", "/")
                    image_files.append(image_path)

    print(f"Found {len(image_files)} images to process:")
    for img in image_files:
        print(f"- {img}")

    for image_path in image_files:
        image = read_gray(image_path)
        convert_to_3d(image)




