# Laser Beam Centroid Detection
This project provides a set of techniques for detecting the centroid of a laser beam in images. It leverages superpixel-based segmentation methods (Felzenszwalb, SLIC, Quickshift) to segment an image and then calculates the centroid with different approaches, offering insights into performance and accuracy.
## Features
- **Superpixel Segmentation**
  Demonstrates how to apply several segmentation algorithms to help isolate the laser beam region in images.
- **Multiple Centroid Detection Methods**
  Includes implementations using OpenCV morphological operations and Scikit-Image CCL, allowing for easy comparison of computation time and detection accuracy.
- **Performance Evaluation**
  Measures the execution time of each approach for benchmarking purposes.

## Installation
1.  **Clone the repository**
``` bash
   git clone https://github.com/greg-ogs/Centroid-for-spot-laser.git
```
2.  **Navigate to the project directory**
``` bash
   cd Centroid-for-spot-laser
```
3.  **Install dependencies**
   Make sure you have Python 3.7+ environment, then run:
``` bash
   pip install --no-cache-dir -r requirements.txt
```
>Alternatively, you can build a dev container using the Dockerfile.

### Docker Instructions
1. **Build the Docker image**
``` bash
   docker build -t laser-beam-centroid .
```
2.  **Run the Docker container**
``` bash
   docker run -it --rm laser-beam-centroid
```
This will create a reproducible environment with all required dependencies installed.
## Usage
1. **Prepare your input image**
   Ensure you have all the directories / classes in the image directory
2. **Run the script**
``` bash
   python main.py
```
This will apply the different segmentation algorithms and centroid methods, then print out the computation times for each approach.

3. **Modify parameters**
   Feel free to customize paths, algorithm parameters, or other settings within the code to suit your needs.



