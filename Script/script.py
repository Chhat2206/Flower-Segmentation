# RUDIMENTARY IMPLEMENTATION FOR TESTING PURPOSES, THIS IS NOT PRODUCTION CODE

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_noise_reduction(image, method='gaussian', kernel_size=5):
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    else:
        raise ValueError("Unknown noise reduction method.")

def threshold_image(image, method='otsu'):
    if method == 'otsu':
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError("Unknown thresholding method.")
    return binary_image

def process_image(input_path, output_path):
    # Read image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Image not found.")

    # Convert to grayscale
    gray_image = convert_to_grayscale(image)

    # Noise reduction
    smooth_image = apply_noise_reduction(gray_image)

    # Thresholding/Segmentation
    binary_image = threshold_image(smooth_image)

    # Save the binary image
    cv2.imwrite(output_path, binary_image)

    # Optionally, display the result
    plt.imshow(binary_image, cmap='gray')
    plt.show()

# Assuming the images are stored in directories named 'easy', 'medium', and 'hard'
input_folders = ['easy', 'medium', 'hard']
output_folders = ['output/easy', 'output/medium', 'output/hard']

for difficulty_level in input_folders:
    input_dir = os.path.join('input-images', difficulty_level)
    output_dir = os.path.join('output', difficulty_level)
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_image(input_path, output_path)
