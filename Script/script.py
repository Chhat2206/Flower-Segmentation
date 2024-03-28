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
    if image.ndim > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
def apply_morphological_transformations(image, operation='open', kernel_size=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'open':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'close':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError("Unknown morphological operation.")


def apply_morphological_operations(image, kernel_size=3, operations=['erosion', 'dilation']):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if 'erosion' in operations:
        image = cv2.erode(image, kernel, iterations=10)

    if 'dilation' in operations:
        image = cv2.dilate(image, kernel, iterations=10)

    return image

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

def kmeans_segmentation(image, k=2, iterations=10):
    # Convert to a floating-point precision as required by cv2.kmeans
    Z = image.reshape((-1, 3)).astype(np.float32)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, 1.0)
    _, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to uint8 and reshape to the image shape
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented_image = res.reshape((image.shape))

    # Assuming the flower is the lighter object, get the cluster with higher intensity
    brightness = np.sum(center, axis=1)
    flower_cluster = np.argmax(brightness)

    # Create a binary mask where the flower cluster has value 1, the rest 0
    binary_mask = (label == flower_cluster).reshape((image.shape[0], image.shape[1])).astype(np.uint8) * 255

    return binary_mask


def advanced_contrast_stretch(image, lower_percentile, upper_percentile):
    # Compute the lower and upper percentile values
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)

    # Saturate values below the lower bound to 0 and above the upper bound to 255
    image = np.clip(image, lower_bound, upper_bound)

    # Scale the intensity values to the full range [0, 255]
    image = (image - lower_bound) / (upper_bound - lower_bound) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


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


def canny_edge_detection(image, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged


def find_and_draw_contours(image, edged):
    # Find contours in the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    return image

def process_and_compare_image(input_path, ground_truth_path):

    # Input image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Image not found at the path.")

# Image Processing Pipeline
    # Apply bilateral filtering for noise reduction
    bilateral_filtered_image = apply_bilateral_filter(image)

    stretched_image = advanced_contrast_stretch(bilateral_filtered_image, lower_percentile=5, upper_percentile=95)

    # Adjust the gain and gamma
    # gain_adjusted_image = adjust_gain(bilateral_filtered_image, 1)
    # gamma_corrected_image = gamma_correction(gain_adjusted_image, 1)

    # Convert to grayscale
    gray_image = convert_to_grayscale(stretched_image)

    # Apply Canny Edges
    canny_edges = canny_edge_detection(gray_image)
    contoured_image = find_and_draw_contours(image.copy(), canny_edges)

# Otsu Segmentation
    # Apply thresholding as the final step
    final_segmentation = threshold_image(contoured_image, method='otsu')

    # Apply Morphology
    # Apply morphological operations and visualize
    eroded_image = apply_morphological_transformations(final_segmentation, 'open')
    dilated_image = apply_morphological_transformations(eroded_image, 'close')


# Comparison with ground Truth
    # Process the ground truth
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    if ground_truth is None:
        raise ValueError("Ground truth image not found at the path.")

    # Make the ground truth binary and invert it
    _, binary_ground_truth = cv2.threshold(ground_truth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_ground_truth = invert_colors(binary_ground_truth)

    # Calculate mIoU score
    miou_score = calculate_miou(dilated_image // 255, inverted_ground_truth // 255)

    # Collect images for comparison
    images = [
        image, bilateral_filtered_image, gray_image, eroded_image,
        dilated_image, final_segmentation, inverted_ground_truth
    ]
    descriptions = [
        "Original Image", "Bilateral Noise Reduction",
        "Grayscale", "Erosion (Open)", "Dilation (Close)", "Final Segmentation (Otsu)", "Inverted Ground Truth"
    ]

    # # Collect images for comparison
    # images = [
    #     image, bilateral_filtered_image, gain_adjusted_image, gamma_corrected_image, gray_image, eroded_image,
    #     dilated_image, final_segmentation, inverted_ground_truth
    # ]
    # descriptions = [
    #     "Original Image", "Bilateral Noise Reduction", "Gain Adjustment", "Gamma Correction",
    #     "Grayscale", "Erosion (Open)", "Dilation (Close)", "Final Segmentation (Otsu)", "Inverted Ground Truth"
    # ]

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
