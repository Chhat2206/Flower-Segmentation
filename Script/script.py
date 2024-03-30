import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def invert_colors(image):
    return cv2.bitwise_not(image)

def calculate_miou(prediction, target, target_original_shape):
    # Resize target to match prediction's shape if they differ
    if prediction.shape != target_original_shape:
        target_resized = cv2.resize(target, (prediction.shape[1], prediction.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        target_resized = target

    intersection = np.logical_and(target_resized, prediction)
    union = np.logical_or(target_resized, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# New function to convert to HSV and split

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

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def create_mask_from_contour(shape, contour):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    return mask

def apply_kmeans(image, K=2):
    # If the image is grayscale, convert it back to BGR
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Reshape the image to a 2D array of pixels and 3 color values (HSV)
    pixel_values = hsv_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Create a mask based on the labels
    mask = labels.reshape(hsv_image.shape[:2])

    # Choose which label corresponds to the flower
    # This can be improved with a more complex logic
    flower_label = 1 if np.sum(centers[0]) < np.sum(centers[1]) else 0

    # Create a binary mask where the flower label is white, and the rest is black
    flower_mask = np.where(mask == flower_label, 255, 0).astype('uint8')

    return flower_mask


# https://publisher.uthm.edu.my/periodicals/index.php/eeee/article/view/441
# not sure it does anything
def apply_median_filter(image, kernel_size=5):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    return cv2.medianBlur(image, kernel_size)


def process_and_compare_image(input_path, ground_truth_path):
    # Load the image
    image = cv2.imread(input_path)

    # Proceed with bilateral filtering
    bilateral_filtered_image = apply_bilateral_filter(image)

    # Convert image to grayscale
    grayscale_image = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)

    # Apply median filtering
    median_filtered_image = apply_median_filter(grayscale_image, kernel_size=9)

    # Convert grayscale to BGR before applying K-means
    median_filtered_bgr = cv2.cvtColor(median_filtered_image, cv2.COLOR_GRAY2BGR)

    # Apply K-means clustering for segmentation on BGR image
    kmeans_result = apply_kmeans(median_filtered_bgr)

    # Find the largest contour in the K-means result
    largest_contour = find_largest_contour(kmeans_result)

    # Draw contours on the image for visualization
    contour_image = cv2.drawContours(image.copy(), [largest_contour], -1, (0, 255, 0), 2)

    # Create an ROI mask from the largest contour
    roi_mask = create_mask_from_contour(kmeans_result.shape, largest_contour)

    # Visualize the ROI mask
    roi_mask_vis = roi_mask.copy()
    contour_idx = 0  # Index of the largest contour
    cv2.drawContours(roi_mask_vis, [largest_contour], contour_idx, (255), 2)

    # Refine the K-means result using the ROI mask
    refined_kmeans_result = cv2.bitwise_and(kmeans_result, kmeans_result, mask=roi_mask)

    # Morphological Operations to refine the segmentation further
    morph_result = apply_morphological_operations(refined_kmeans_result)
    # Process the ground truth
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    if ground_truth is None:
        raise ValueError("Ground truth image not found at the path.")

    _, binary_ground_truth = cv2.threshold(ground_truth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert colors if necessary
    inverted_ground_truth = invert_colors(binary_ground_truth)

    # Calculate mIoU score
    miou_score = calculate_miou(morph_result // 255, inverted_ground_truth // 255, inverted_ground_truth.shape)

    # Collect images for comparison
    images = [image, bilateral_filtered_image, grayscale_image, median_filtered_image, kmeans_result, roi_mask_vis,
              contour_image, refined_kmeans_result, inverted_ground_truth, morph_result]
    descriptions = ["Original Image", "Bilateral Filtered", "Grayscale", "Median Filtered", "K-Means", "ROI Mask",
                    "Contours", "Refined K-Means", "Inverted Ground Truth", "Morphology"]

    return miou_score, images, descriptions


def process_images_and_compare(directory_paths, ground_truth_directory_paths):
    miou_scores = {}
    total_images = 9  # Total sets of images to process
    steps_per_set = 12  # Steps per set

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
