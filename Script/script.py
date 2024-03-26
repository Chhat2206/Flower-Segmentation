import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def adaptive_gaussian_blur(image):
    # Check if the image is grayscale or color
    if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale image
        gray_image = image
    else:  # Color image
        gray_image = convert_to_grayscale(image)

    # Calculate the Laplacian variance on the grayscale image
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    # Adjust the kernel size based on the variance
    if laplacian_var < 300:  # Low variance, image is likely smooth
        kernel_size = (9, 9)
    elif laplacian_var < 600:  # Moderate variance
        kernel_size = (7, 7)
    else:  # High variance, preserve more details
        kernel_size = (5, 5)

    # Apply Gaussian Blur to the original image, not the grayscale version
    return cv2.GaussianBlur(image, kernel_size, 0)

def apply_otsu_threshold(image, method='otsu'):
    if method == 'otsu':
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image
    else:
        raise ValueError("Unknown thresholding method.")

def invert_colors(image):
    return cv2.bitwise_not(image)


def calculate_miou(prediction, target):
    # Ensure both images have the same number of dimensions
    if len(target.shape) == 2:  # Target is grayscale
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2GRAY)
    elif len(target.shape) == 3:  # Target is color
        prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)

    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


# New function to convert to HSV and split
def convert_to_hsv_and_split(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    return h, s, v

# Placeholder for morphological transformations
def apply_morphological_transformations(image, operation='open', kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'open':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError("Unknown morphological operation.")

# Function to refine contours and extract the flower with a white background
def refine_contours_and_extract_roi(image, edges):
    # Find contours based on edges detected
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assuming the largest contour is the flower
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Create a mask for the largest contour
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 50, 10)
        # Create a white background
        white_background = np.full(image.shape, 255, dtype=np.uint8)
        # Apply the mask to the flower and the background
        flower = cv2.bitwise_and(image, image, mask=mask)
        white_background = cv2.bitwise_and(white_background, white_background, mask=cv2.bitwise_not(mask))
        # Combine the flower with the white background
        final_image = cv2.add(flower, white_background)
        return final_image
    else:
        return image

def process_image_and_calculate_iou(input_path, ground_truth_path):
    # Read the images
    image = cv2.imread(input_path)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    # Check if images were successfully loaded
    if image is None or ground_truth is None:
        raise ValueError("Image or ground truth not found.")

    # Convert the image to grayscale
    gray_image = convert_to_grayscale(image)
    # Apply adaptive Gaussian blur
    noise_reduced_image = adaptive_gaussian_blur(gray_image)
    # Apply edge detection
    edges = cv2.Canny(noise_reduced_image, 50, 150)
    # Extract ROI with a white background
    roi_image = refine_contours_and_extract_roi(image, edges)
    # Convert the ground truth to binary and invert the colors
    binary_ground_truth = apply_otsu_threshold(ground_truth)
    inverted_binary_ground_truth = invert_colors(binary_ground_truth)
    # Calculate mIoU score
    miou_score = calculate_miou(roi_image, inverted_binary_ground_truth)

    # Prepare images and descriptions for plotting
    images = [
        image,
        gray_image,
        noise_reduced_image,
        edges,
        roi_image,
        binary_ground_truth,
        inverted_binary_ground_truth
    ]
    descriptions = [
        "Original Image",
        "Grayscale Image",
        "Adaptive Gaussian Blur",
        "Edge Detection",
        "Region of Interest (ROI)",
        "Binary Ground Truth",
        "Inverted Binary Ground Truth"
    ]

    # Plot the results
    fig, axs = plt.subplots(1, len(images), figsize=(20, 5))
    for ax, img, desc in zip(axs, images, descriptions):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
        ax.set_title(desc)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return miou_score, images, descriptions

# Function to refine contours and extract the ROI with a white background
def refine_contours_and_extract_roi(image, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # If no contours are detected
    if not contours:
        return image
    # Assuming the largest contour in the image is the flower
    largest_contour = max(contours, key=cv2.contourArea)
    # Create a mask for the largest contour
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    # Create a white background the same size as the image
    white_background = np.full(image.shape, 255, dtype=np.uint8)
    # Apply the mask to the image to extract the flower
    flower = cv2.bitwise_and(image, image, mask=mask)
    # Combine the extracted flower with the white background
    roi_image = cv2.bitwise_and(white_background, white_background, mask=cv2.bitwise_not(mask))
    roi_image += flower
    return roi_image

def process_multiple_images_and_calculate_iou(directory_paths, ground_truth_directory_paths):
    miou_scores = {}
    total_images = 9  # Total sets of images to process
    steps_per_set = 7  # Steps per set

    # Adjust here for overall plotting
    fig, axs = plt.subplots(total_images, steps_per_set, figsize=(20, total_images * 2.5))

    current_image_index = 0  # To keep track of which row we're on

    for difficulty in ['easy', 'medium', 'hard']:
        for i in range(1, 4):  # Assuming there are 3 images per difficulty level
            ground_truth_path = f'{ground_truth_directory_paths[difficulty]}/{difficulty}_{i}.png'
            modified_ground_truth_path = f'{ground_truth_directory_paths[difficulty]}/{difficulty}_{i}_modified.png'
            inverted_ground_truth_path = f'{ground_truth_directory_paths[difficulty]}/{difficulty}_{i}_inverted.png'

            # Proceed with color modification and inversion as before

            input_path = f'{directory_paths[difficulty]}/{difficulty}_{i}.jpg'
            print(f"Processing {input_path} with inverted ground truth {inverted_ground_truth_path}")

            try:
                score, images, descriptions = process_image_and_calculate_iou(input_path, inverted_ground_truth_path)
                miou_scores[input_path] = score

                # Plot each step in the process for the current set
                for step_index, (img, desc) in enumerate(zip(images, descriptions)):
                    ax = axs[current_image_index, step_index]
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
                    ax.set_title(desc, fontsize=9)
                    ax.axis('off')

                current_image_index += 1  # Move to the next row for the next set of images

            except Exception as e:
                print(f"Error processing {input_path}: {e}")

    for path, score in miou_scores.items():
        print(f"mIoU for {path}: {score}")

    plt.tight_layout()
    plt.show()

# Define your directory paths
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

# Call the main function
process_multiple_images_and_calculate_iou(directory_paths, ground_truth_directory_paths)
