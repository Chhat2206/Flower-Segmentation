import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

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

    # Apply morphological transformation to binary_image or as needed
    morphologically_transformed_image = apply_morphological_transformations(binary_image, 'open', 5)

    score, _ = ssim(morphologically_transformed_image, binary_ground_truth, full=True)
    images = [image, gray_image, noise_reduced_image, additionally_blurred_image, binary_image, binary_ground_truth, morphologically_transformed_image]
    descriptions = [
        "Original Image",
        "Grayscale Conversion",
        "Noise Reduction (Gaussian Blur)",
        "Additional Blur",
        "Binary Threshold (Otsu's Method)",
        "Inverted Ground Truth",
        "Morphological Transformation"
    ]
    return score, images, descriptions

def modify_colors(image_path, modified_image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image from BGR to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the color ranges for red, yellow and black
    red_lower = np.array([100, 0, 0], dtype="uint8")
    red_upper = np.array([255, 100, 100], dtype="uint8")

    yellow_lower = np.array([0, 100, 100], dtype="uint8")
    yellow_upper = np.array([100, 255, 255], dtype="uint8")

    black_lower = np.array([0, 0, 0], dtype="uint8")
    black_upper = np.array([50, 50, 50], dtype="uint8")

    # Create masks for red, yellow and black colors
    red_mask = cv2.inRange(image_rgb, red_lower, red_upper)
    yellow_mask = cv2.inRange(image_rgb, yellow_lower, yellow_upper)
    black_mask = cv2.inRange(image_rgb, black_lower, black_upper)

    # Create a mask for non-black (white and yellow) areas
    non_black_mask = cv2.bitwise_or(red_mask, yellow_mask)

    # Modify the colors
    image_rgb[non_black_mask == 255] = [255, 255, 255]  # Change red and yellow to white
    image_rgb[yellow_mask == 255] = [0, 0, 0]  # Change yellow to black
    # Black areas remain unchanged

    # Convert the image from RGB to BGR format
    modified_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Save the modified image
    modified_image_path = image_path.replace('.png', '_modified.png')
    cv2.imwrite(modified_image_path, modified_image)

    return modified_image_path

def process_images_and_compare(directory_paths, ground_truth_directory_paths):
    ssim_scores = {}

    for difficulty in ['easy', 'medium', 'hard']:
        for i in range(1, 4):  # Assuming there are 3 images per difficulty level
            ground_truth_path = f'{ground_truth_directory_paths[difficulty]}/{difficulty}_{i}.png'
            modified_ground_truth_path = f'{ground_truth_directory_paths[difficulty]}/{difficulty}_{i}_modified.png'
            inverted_ground_truth_path = f'{ground_truth_directory_paths[difficulty]}/{difficulty}_{i}_inverted.png'

            # First, modify the colors of the ground truth image
            modify_colors(ground_truth_path, modified_ground_truth_path)

            # Then, invert the colors of the modified ground truth image
            ground_truth = cv2.imread(modified_ground_truth_path, cv2.IMREAD_GRAYSCALE)
            inverted_ground_truth = invert_colors(ground_truth)
            cv2.imwrite(inverted_ground_truth_path, inverted_ground_truth)

            input_path = f'{directory_paths[difficulty]}/{difficulty}_{i}.jpg'
            print(f"Processing {input_path} with inverted ground truth {inverted_ground_truth_path}")
            try:
                score, images, descriptions = process_and_compare_image(input_path, inverted_ground_truth_path)
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