import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim

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
        return binary_image
    else:
        raise ValueError("Unknown thresholding method.")

def invert_colors(image):
    # Invert the image colors
    return cv2.bitwise_not(image)

def process_and_compare_image(input_path, ground_truth_path):
    # Read the input and ground truth images
    image = cv2.imread(input_path)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    if image is None or ground_truth is None:
        raise ValueError("Image or ground truth not found.")

    # Invert the ground truth colors and ensure it is binary
    inverted_ground_truth = invert_colors(ground_truth)
    _, binary_ground_truth = cv2.threshold(inverted_ground_truth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Processing pipeline
    gray_image = convert_to_grayscale(image)
    smooth_image = apply_noise_reduction(gray_image)
    binary_image = threshold_image(smooth_image)

    # Display both images for visual comparison
    plt.figure(figsize=(10, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(binary_image, cmap='gray')
    plt.title("Processed Binary Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(binary_ground_truth, cmap='gray')
    plt.title("Inverted Ground Truth")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Compute the Structural Similarity Index (SSIM) between processed image and inverted ground truth
    score, _ = ssim(binary_image, binary_ground_truth, full=True)
    print(f"SSIM between processed image and inverted ground truth: {score}")

# Paths
input_path = 'input-images/easy/easy_1.jpg'
base_name = os.path.basename(input_path).replace('.jpg', '_binary.jpg')
ground_truth_path = 'ground_truths/easy/easy_1.png'

# Process the image and compare with inverted ground truth
process_and_compare_image(input_path, ground_truth_path)
