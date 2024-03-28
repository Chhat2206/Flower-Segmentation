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


def apply_watershed_segmentation(image):
    # Convert to grayscale and apply a binary threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal (optional, based on your image)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply watershed
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Mark boundaries with red

    return image

def process_and_compare_image(input_path, ground_truth_path):
    # Read the image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Image not found at the path.")

    # Apply noise reduction
    noise_reduced_image = apply_noise_reduction(image)

    # Convert to grayscale
    gray_image = convert_to_grayscale(noise_reduced_image)  # Use the noise-reduced image to convert to grayscale

    # Apply erosion and visualize
    eroded_image = cv2.erode(gray_image, np.ones((3, 3), np.uint8), iterations=1)  # Adjust iterations as needed

    # Apply dilation and visualize
    dilated_image = cv2.dilate(eroded_image, np.ones((3, 3), np.uint8), iterations=1)  # Adjust iterations as needed

    # Apply Otsu's thresholding as the final step on the dilated image
    final_segmentation = threshold_image(dilated_image, method='otsu')

    # Process the ground truth
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    if ground_truth is None:
        raise ValueError("Ground truth image not found at the path.")

    # Make the ground truth binary using Otsu's method and invert it
    _, binary_ground_truth = cv2.threshold(ground_truth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_ground_truth = invert_colors(binary_ground_truth)

    # Calculate mIoU score
    miou_score = calculate_miou(final_segmentation // 255, inverted_ground_truth // 255)

    # Collect images for comparison, including erosion and dilation steps
    images = [
        image, noise_reduced_image, gray_image, eroded_image, dilated_image,
        final_segmentation, inverted_ground_truth
    ]
    descriptions = [
        "Original Image", "Noise Reduction", "Grayscale", "Erosion",
        "Dilation", "Final Segmentation (Otsu)", "Inverted Ground Truth"
    ]

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
