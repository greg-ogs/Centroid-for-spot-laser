# Laser Beam Centroid Detection
This project provides a set of techniques for detecting the centroid of a laser beam in images. It leverages superpixel-based segmentation methods (Felzenszwalb, SLIC, Quickshift) to segment an image and then calculates the centroid with different approaches, offering insights into performance and accuracy.
## Features
- **Superpixel Segmentation**
  Demonstrates how to apply several segmentation algorithms to help isolate the laser beam region in images.
- **Multiple Centroid Detection Methods**
  Includes implementations using OpenCV and Scikit-Image, allowing for easy comparison of computation time and detection accuracy.
- **Performance Evaluation**
  Measures the execution time of each approach for benchmarking purposes.

## Installation
1. **Clone the repository**
``` bash
   git clone https://github.com/YourUsername/your-repo.git
```
1. **Navigate to the project directory**
``` bash
   cd your-repo
```
1. **Install dependencies**
   Make sure you have Python 3.7+ installed, then run:
``` bash
   pip install --no-cache-dir -r requirements.txt
```
## Usage
1. **Prepare your input image**
   Ensure you have an image containing a laser spot to analyze.
2. **Run the script**
``` bash
   python main.py
```
This will apply the different segmentation algorithms and centroid methods, then print out the computation times for each approach.
1. **Modify parameters**
   Feel free to customize paths, algorithm parameters, or other settings within the code to suit your needs.

## Docker Instructions
1. **Build the Docker image**
``` bash
   docker build -t laser-beam-centroid .
```
1. **Run the Docker container**
``` bash
   docker run -it --rm laser-beam-centroid
```
This will create a reproducible environment with all required dependencies installed.

