import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_noise_reduction(image, method='gaussian', kernel_size=7):
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 5)
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

def calculate_miou(prediction, target):
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

def process_and_compare_image(input_path, ground_truth_path):
    # Read the image
    image = cv2.imread(input_path)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    if image is None or ground_truth is None:
        raise ValueError("Image or ground truth not found.")

    # Apply noise reduction before converting to grayscale
    noise_reduced_image = apply_noise_reduction(image)  # Applying noise reduction on the original image
    gray_image = convert_to_grayscale(noise_reduced_image)  # Now converting to grayscale

    # Continue with your existing pipeline...
    additionally_blurred_image = apply_additional_blur(gray_image)
    binary_image = threshold_image(additionally_blurred_image)

    # Apply morphological transformations as needed
    morphologically_transformed_image = apply_morphological_transformations(binary_image, 'open', 1)

    # Invert the colors of the ground truth image to get the binary mask
    inverted_ground_truth = invert_colors(ground_truth)
    _, binary_ground_truth = cv2.threshold(inverted_ground_truth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate mIoU score
    miou_score = calculate_miou(morphologically_transformed_image // 255, binary_ground_truth // 255)

    # Collect images for comparison
    images = [
        image,
        noise_reduced_image,  # This is the noise-reduced color image
        gray_image,
        additionally_blurred_image,
        binary_image,
        binary_ground_truth,
        morphologically_transformed_image
    ]
    descriptions = [
        "Original Image",
        "Noise Reduction (Original)",
        "Grayscale Conversion",
        "Additional Blur",
        "Binary Threshold (Otsu's Method)",
        "Inverted Ground Truth",
        "Morphological Transformation"
    ]

    return miou_score, images, descriptions

def process_images_and_compare(directory_paths, ground_truth_directory_paths):
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
                score, images, descriptions = process_and_compare_image(input_path, inverted_ground_truth_path)
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
process_images_and_compare(directory_paths, ground_truth_directory_paths)
