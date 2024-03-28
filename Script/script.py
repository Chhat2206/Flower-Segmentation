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

def equalize_histogram(image):
    return cv2.equalizeHist(image)

def threshold_image(image, method='otsu'):
    if method == 'otsu':
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image
    else:
        raise ValueError("Unknown thresholding method.")

def invert_colors(image):
    return cv2.bitwise_not(image)

def adaptive_threshold_image(image):
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=1,  # Size of a pixel neighborhood that is used to calculate a threshold value
        C=2  # Constant subtracted from the mean or weighted mean
    )

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
def apply_morphological_operations(image, close_kernel_size=3, open_kernel_size=3):
    # Create structuring elements for morphological operations
    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)

    # Closing: Dilation followed by Erosion to fill holes
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, close_kernel)

    # Opening: Erosion followed by Dilation to remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, open_kernel)

    return opening

def segment_flower_using_color(image):
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the color range for yellow flowers
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a mask for the yellow color
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

    return clean_mask

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):

    """

    Apply bilateral filtering to reduce noise while preserving edges.



    Parameters:

    - image: The input image

    - d: Diameter of each pixel neighborhood used during filtering

    - sigma_color: Value for filter sigma in the color space

    - sigma_space: Value for filter sigma in the coordinate space



    Returns:

    - The filtered image

    """

    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def adjust_gain(image, gain=1.0):
    """
    Adjusts the gain (brightness) of an image.

    Parameters:
    - image: Input image
    - gain: Factor by which to multiply pixel values

    Returns:
    - The brightness adjusted image
    """
    # Convert to float to avoid clipping during multiplication
    f_image = image.astype(np.float32)
    # Multiply the image by the gain factor
    f_image = cv2.multiply(f_image, np.array([gain]))
    # Clip values to the range [0, 255] and convert back to uint8
    adjusted_image = np.clip(f_image, 0, 255).astype(np.uint8)
    return adjusted_image

def gamma_correction(image, gamma=1.0):
    """
    Performs gamma correction on an image.

    Parameters:
    - image: Input image
    - gamma: Gamma value for correction

    Returns:
    - The gamma corrected image
    """
    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype(np.uint8)

    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def apply_kmeans(image, K=2):
    # Convert image into a feature vector
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8 bit values
    centers = np.uint8(centers)

    # Map labels to center values
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    # Assuming the flower is brighter, find which cluster is brighter
    brightness = np.sum(centers, axis=1)
    flower_cluster = np.argmax(brightness)

    # Create a mask where the brighter cluster is white, and the other is black
    mask = np.where(labels == flower_cluster, 255, 0)
    mask = mask.astype(np.uint8)
    mask = mask.reshape((image.shape[0], image.shape[1]))

    return mask

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    updated_image = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return updated_image

def process_and_compare_image(input_path, ground_truth_path):
    # Load the image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Image not found at the path.")

    # Noise Reduction
    preprocessed_image = apply_noise_reduction(image)

    # Bilateral filtering
    bilateral_filtered_image = apply_bilateral_filter(preprocessed_image)

    # Apply K-means clustering for segmentation
    kmeans_result = apply_kmeans(bilateral_filtered_image)

    # Morphological Operations to refine the segmentation
    morph_result = apply_morphological_operations(kmeans_result)

    # Process the ground truth
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    if ground_truth is None:
        raise ValueError("Ground truth image not found at the path.")

    _, binary_ground_truth = cv2.threshold(ground_truth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert colors if necessary
    inverted_ground_truth = invert_colors(binary_ground_truth)

    # Calculate mIoU score
    miou_score = calculate_miou(morph_result // 255, inverted_ground_truth // 255)

    # Collect images for comparison
    images = [image, bilateral_filtered_image, kmeans_result, inverted_ground_truth, morph_result]
    descriptions = ["Original Image", "Preprocessed Image", "K-means Result", "Inverted Ground Truth", "Morphology"]

    return miou_score, images, descriptions


def process_images_and_compare(directory_paths, ground_truth_directory_paths):
    miou_scores = {}
    total_images = 9  # Total sets of images to process
    steps_per_set = 9  # Steps per set

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
