import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_noise_reduction(image, method='gaussian', kernel_size=5):
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    else:
        raise ValueError("Unknown noise reduction method.")

def apply_additional_blur(image, kernel_size=9):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def threshold_image(image, method='otsu'):
    if method == 'otsu':
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image
    else:
        raise ValueError("Unknown thresholding method.")

def invert_colors(image):
    return cv2.bitwise_not(image)

def process_and_compare_image(input_path, ground_truth_path):
    image = cv2.imread(input_path)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    if image is None or ground_truth is None:
        raise ValueError("Image or ground truth not found.")

    inverted_ground_truth = invert_colors(ground_truth)
    _, binary_ground_truth = cv2.threshold(inverted_ground_truth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    gray_image = convert_to_grayscale(image)
    noise_reduced_image = apply_noise_reduction(gray_image)
    additionally_blurred_image = apply_additional_blur(noise_reduced_image)
    binary_image = threshold_image(additionally_blurred_image)

    score, _ = ssim(binary_image, binary_ground_truth, full=True)
    images = [image, gray_image, noise_reduced_image, additionally_blurred_image, binary_image, binary_ground_truth]
    descriptions = [
        "Original Image",
        "Grayscale Conversion",
        "Noise Reduction (Gaussian Blur)",
        "Additional Blur",
        "Binary Threshold (Otsu's Method)",
        "Inverted Ground Truth"
    ]
    return score, images, descriptions

def process_images_and_compare(directory_paths, ground_truth_directory_paths):
    ssim_scores = {}
    for difficulty in ['easy', 'medium', 'hard']:
        for i in range(1, 4):  # Assuming there are 3 images per difficulty level
            input_path = f'{directory_paths[difficulty]}/{difficulty}_{i}.jpg'
            ground_truth_path = f'{ground_truth_directory_paths[difficulty]}/{difficulty}_{i}.png'
            print(f"Processing {input_path} with ground truth {ground_truth_path}")
            try:
                score, images, descriptions = process_and_compare_image(input_path, ground_truth_path)
                ssim_scores[input_path] = score
                # Displaying images with descriptions for each step
                fig, axs = plt.subplots(1, len(images), figsize=(20, 5))
                for ax, img, desc in zip(axs, images, descriptions):
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
                    ax.text(0.5, -0.1, desc, fontsize=9, ha='center', transform=ax.transAxes)
                    ax.axis('off')
                plt.show()
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

    # Print SSIM scores
    for path, score in ssim_scores.items():
        print(f"SSIM for {path}: {score}")

directory_paths = {
    'easy': 'input-images/easy',
    'medium': 'input-images/medium',
    'hard': 'input-images/hard'
}
ground_truth_directory_paths = {
    'easy': 'ground_truths/easy',
    'medium': 'ground_truths/medium',
    'hard': 'ground_truths/hard'
}

process_images_and_compare(directory_paths, ground_truth_directory_paths)
