import os

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

def apply_kmeans(image, K=2):
    # Check if the image is grayscale. If so, convert it to a 3-channel format.
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convert image into a feature vector
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, (centers) = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Compute the mean intensity of each cluster
    mean_intensity = np.mean(centers, axis=1)
    flower_label = np.argmax(mean_intensity)

    # Construct the mask based on the identified flower cluster
    mask = np.zeros(labels.shape, dtype=np.uint8)
    mask[labels == flower_label] = 255
    mask = mask.reshape((image.shape[0], image.shape[1]))

    return mask, image


# https://publisher.uthm.edu.my/periodicals/index.php/eeee/article/view/441
# not sure it does anything
def apply_median_filter(image, kernel_size=5):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")
    return cv2.medianBlur(image, kernel_size)


def apply_watershed(image, segmentation_mask):
    # Convert segmentation mask to binary format
    _, binary_mask = cv2.threshold(segmentation_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Noise reduction in the binary mask
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Sure background area using dilation
    sure_bg = cv2.dilate(opening, np.ones((9, 9), np.uint8), iterations=3)

    # Finding sure foreground area using distance transform and thresholding
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)

    # Finding unknown region by subtracting sure foreground from sure background
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that the background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply the Watershed algorithm
    markers = cv2.watershed(image, markers)

    # Create the segmentation result
    segmentation = np.zeros_like(binary_mask)
    segmentation[markers == -1] = 255  # Edge marking
    segmentation[markers > 1] = 255  # Object marking

    # Post-processing to fill holes and remove noise
    segmentation = cv2.morphologyEx(segmentation, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    return segmentation


def pipeline(input_path, ground_truth_path):
    # Load the image
    image = cv2.imread(input_path)

    # Proceed with bilateral filtering
    bilateral_filtered_image = apply_bilateral_filter(image)

    # Convert image to grayscale
    grayscale_image = cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2GRAY)

    # Apply median filtering
    median_filtered_image = apply_median_filter(grayscale_image)

    # Apply K-means clustering for segmentation
    kmeans_mask, image_for_watershed = apply_kmeans(median_filtered_image)

    # Apply Watershed algorithm to refine segmentation
    watershed_result = apply_watershed(image_for_watershed, kmeans_mask)

    # Morphological Operations to further refine the segmentation
    morph_result = apply_morphological_operations(watershed_result)

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
    images = [image, bilateral_filtered_image, grayscale_image, median_filtered_image, kmeans_mask, watershed_result,
              morph_result, inverted_ground_truth]
    descriptions = ["Original Image", "Bilateral Filtered", "Grayscale", "Median Filtered", "K-Means Result",
                    "Watershed Result", "Morphology Result (Final)", "Inverted Ground Truth"]

    return miou_score, images, descriptions

def save_images(directory_paths, ground_truth_directory_paths, output_directory_paths):

    for difficulty in ['easy', 'medium', 'hard']:
        output_directory = output_directory_paths[difficulty]
        os.makedirs(output_directory, exist_ok=True)  # Create output directory if not exists

        current_image_index = 0  # To keep track of which row we're on

        for i in range(1, 4):  # Assuming there are 3 images per difficulty level
            input_path = f'{directory_paths[difficulty]}/{difficulty}_{i}.jpg'
            ground_truth_path = f'{ground_truth_directory_paths[difficulty]}/{difficulty}_{i}.png'
            output_path = f"{output_directory}/{difficulty}_{i}.png"

            try:
                _, _, _ = pipeline(input_path, ground_truth_path)
                # Only save the morph_result image
                _, images, descriptions = pipeline(input_path, ground_truth_path)
                morph_result = images[-1]  # Get the last image, which is the morph_result

                # Save the morph_result image
                cv2.imwrite(output_path, morph_result)

            except Exception as e:
                print(f"Error processing {input_path}: {e}")

    print("Output images saved successfully.")

def display_images(directory_paths, ground_truth_directory_paths):
    miou_scores = {}
    total_images = 9  # Total sets of images to process
    steps_per_set = 9  # Steps per set

    # Adjust here for overall plotting
    fig, axs = plt.subplots(total_images, steps_per_set, figsize=(20, total_images * 2.5))

    current_image_index = 0  # To keep track of which row we're on

    for difficulty in ['easy', 'medium', 'hard']:
        for i in range(1, 4):  # Assuming there are 3 images per difficulty level
            ground_truth_path = f'{ground_truth_directory_paths[difficulty]}/{difficulty}_{i}.png'

            # Proceed with color modification and inversion as before

            input_path = f'{directory_paths[difficulty]}/{difficulty}_{i}.jpg'
            print(f"Processing {input_path} with ground truth {ground_truth_path}")

            try:
                score, images, descriptions = pipeline(input_path, ground_truth_path)
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
output_directory_paths = {
    'easy': 'output/easy',
    'medium': 'output/medium',
    'hard': 'output/hard'
}

# Call the main functions
save_images(directory_paths, ground_truth_directory_paths, output_directory_paths)
display_images(directory_paths, ground_truth_directory_paths)
