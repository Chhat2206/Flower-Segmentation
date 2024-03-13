import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def convert_to_grayscale(image):
    # Convert the input image to grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_noise_reduction(image, method='gaussian', kernel_size=5):
    if method == 'gaussian':
        # Apply Gaussian blur to the image for noise reduction
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    else:
        raise ValueError("Unknown noise reduction method.")


def threshold_image(image, method='otsu'):
    if method == 'otsu':
        # Apply Otsu's thresholding method to the image for segmentation
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError("Unknown thresholding method.")
    return binary_image


def process_image(input_path):
    # Read the input image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Image not found.")

    # Display the original image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Convert the input image to grayscale and display
    gray_image = convert_to_grayscale(image)
    plt.subplot(2, 2, 2)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    # Apply noise reduction to the grayscale image and display
    smooth_image = apply_noise_reduction(gray_image)
    plt.subplot(2, 2, 3)
    plt.imshow(smooth_image, cmap='gray')
    plt.title('Smoothed Image')
    plt.axis('off')

    # Apply thresholding/segmentation to the smoothed image and display
    binary_image = threshold_image(smooth_image)
    plt.subplot(2, 2, 4)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Image (Otsu\'s Method)')
    plt.axis('off')

    # Construct the output path
    output_path = os.path.splitext(input_path)[0] + "_binary.jpg"

    # Save the binary image
    cv2.imwrite(output_path, binary_image)

    # Show the plot
    plt.tight_layout()
    plt.show()


# Input path for the image to be processed
input_path = 'input-images/easy/easy_1.jpg'

# Process the image
process_image(input_path)
