import cv2
from skimage.segmentation import slic, mark_boundaries
import numpy as np
import os

# Read the image
path = 'E:/data/STERE/RGB/'
cnt = 0

for image_name in os.listdir(path):
    cnt += 1
    print(cnt, image_name)

    save_txt_path = 'E:/STERE/region/' + image_name + '.txt'
    save_visualization_path = 'E:/STERE/SLIC/' + image_name
    image_path = path + image_name
    if not os.path.exists(os.path.dirname(save_txt_path)):
         os.makedirs(os.path.dirname(save_txt_path))
    if not os.path.exists(os.path.dirname(save_visualization_path)):
        os.makedirs(os.path.dirname(save_visualization_path))
    print(image_path)
    image = cv2.imread(image_path)

    # Convert the image to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set hyperparameters
    num_segments = 200  # Number of superpixels to segment
    compactness = 5  # Controls the compactness of the superpixel shape

    # Use SLIC algorithm for image segmentation
    segments = slic(image_rgb, n_segments=num_segments, compactness=compactness)

    # Save the labels
    np.savetxt(save_txt_path, segments, fmt='%d')

    # Visualize the segmentation results
    segmented_image = mark_boundaries(image_rgb, segments, color=(1, 1, 1), mode='thick')

    # Convert image data type to uint8
    segmented_image_uint8 = (segmented_image * 255).astype(np.uint8)

    # Save the segmented visualization image
    cv2.imwrite(save_visualization_path, cv2.cvtColor(segmented_image_uint8, cv2.COLOR_RGB2BGR))