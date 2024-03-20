
def process_images_and_compare(directory_paths, ground_truth_directory_paths):
    ssim_scores = {}
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

            input_path = f'{directory_paths[difficulty]}/{difficulty}_{i}.jpg'
            print(f"Processing {input_path} with inverted ground truth {inverted_ground_truth_path}")

            try:
                # Load the ground truth image separately to check for existence
                ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
                if ground_truth is None:
                    raise ValueError("Ground truth image not found.")

                # Proceed with color modification and inversion as before
                modify_colors(ground_truth_path, modified_ground_truth_path)

                # Then, invert the colors of the modified ground truth image
                inverted_ground_truth = invert_colors(cv2.imread(modified_ground_truth_path, cv2.IMREAD_GRAYSCALE))
                cv2.imwrite(inverted_ground_truth_path, inverted_ground_truth)

                # Now process the input image and compare
                score, images, descriptions = process_and_compare_image(input_path, inverted_ground_truth_path)
                ssim_scores[input_path] = score

                # Plot each step in the process for the current set
                for step_index, (img, desc) in enumerate(zip(images, descriptions)):
                    ax = axs[current_image_index, step_index]
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
                    ax.set_title(desc, fontsize=9)
                    ax.axis('off')

                current_image_index += 1  # Move to the next row for the next set of images

            except Exception as e:
                print(f"Error processing {input_path}: {e}")

    for path, score in ssim_scores.items():
        print(f"SSIM for {path}: {score}")

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
