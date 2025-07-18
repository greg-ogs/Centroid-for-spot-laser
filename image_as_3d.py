'''
Created by greg-ogs at 07/16/2025
'''

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
    wire = ax.plot_wireframe(x[::stride, ::stride],
                             y[::stride, ::stride],
                             img[::stride, ::stride],
                             color='red')

    ax.set_xlabel('X')
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')

    ax.view_init(elev=10, azim=280)

    plt.show()

if __name__ == "__main__":
    img_path = "images/image1001.png"
    image = read_gray(img_path)
    convert_to_3d(image)




