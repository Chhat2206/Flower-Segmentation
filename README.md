# Flower Segmentation through Python
This project proposes an efficient algorithm for segmenting flowers from complex backgrounds in images. The algorithm effectively addresses challenges such as pixel noise and foreground-background collisions by leveraging a pipeline of image processing techniques, including bilateral filtering, greyscale conversion, and saturation adjustment. Segmentation is achieved through K-means clustering and watershed algorithms, supplemented by morphological transformations for refinement. Empirical experimentation guides the selection of preprocessing techniques, leading to a robust pipeline for accurate flower segmentation. This work contributes to advancing automated flower segmentation for applications in botany, agriculture, and computer vision. I achieved around 80 to 90% image segmentation success. 

This Python script implements an image segmentation pipeline using bilateral filtering and median filtering to remove noise from the image. The segmentation techniques used include K-means clustering, watershed segmentation, and morphological operations to clean the segmented image. 

## Dependencies:

- Python version 3 and above
- OpenCV
- NumPy
- Matplotlib

### Running the program:
1. Copy the entire src folder into the folder of your choice. 
2. Open that folder in PyCharm as a new project
3. Install the required dependencies listed above. 
   - Often times, a pop-up will ask to automatically install the dependencies for you. If that does not occur, follow the installation file below
4. There will be a ground-truth, input-images and output folder ready for you. Replace the files in this folder exactly with the same folder names, file names, and file formats for the script to run perfectly.

### Manual Installation:

#### Installation through an IDE:
- For the installation of the required dependencies:
    - Open your IDE
    - Go to the Python packages manager
    - Search for the required dependencies and install them

#### Manual Installation of dependencies:
- pip install opencv-python
- pip install matplotlib
- pip install numpy

## Manual Directories Creation
1. Make two directories:
    - "input-images"
    - "ground-truths"
    
    * The `input-images` directory is where you input the flowers to be processed.
        - The `input-images` directory has to be separated into different folders depending on the difficulty of the image.
        
    * The `ground-truths` are the optimal-segmented flower that will be used for comparison with the input-image after processing.
    
    * Both directories must be in the same src folder.
    
```python
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
```
# Notes
- If you want to visualize the process and the GUI, uncomment plt.show() in the display_images function.
