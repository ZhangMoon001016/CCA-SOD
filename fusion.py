import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw
import cv2
import os

def read_image_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip().split() for line in lines]
        image_array = np.array([[int(pixel) for pixel in line] for line in lines])
        # print(image_array)
    return image_array

def get_coordinates_for_class(arr, class_label):  # Get coordinate positions
    coordinates = np.column_stack(np.where(arr == class_label))
    return coordinates

def get_score_by_id(file_path, target_id):
    # Open the file
    with open(file_path, 'r') as file:
        # Read the file line by line
        for line in file:
            # print(line)
            # Split each line, assuming the format is "ID Score"
            parts = line.strip().split()
            current_id, score = parts
            # print(current_id, score, target_id)
            # Check if the current line ID matches the target ID
            if current_id == str(target_id):
                 # Return the corresponding score
                 s = score
    # print(s)
    return s

def apply_color_to_regions(sal1, sal2, regions, score_rgb, score_depth):
    if score_depth < score_rgb:
        for (y, x) in regions:
                # print(x, y)
                sal2[y, x] = sal1[y, x]

    return sal2

RGB_sal = 'E:/data/SIP/BBRF_0/'
# RGBD_sal = 'C:/Users/zhang/Desktop/Experiment/STATE-OF-THE-ART/23/PICR-Net23/PICR-Net/SIP/'
RGBD_sal = 'C:/Users/zhang/Desktop/Experiment/STATE-OF-THE-ART/23/caver-r101d-njudnlpr23/sip/'
# PICR-Net23/PICR-Net/SSD/
# caver-r101d-njudnlpr23/ssd/
RGB_score = 'E:/data/SLIC/SIP/RGB/score/'
d_score = 'E:/data/SLIC/SIP/depth/score/'
label_txt = 'E:/data/SLIC/SIP/txt_slic/'
save_path = 'C:/Users/zhang/Desktop/Experiment/Our_saliencymap/SLIC/SIP/BBRF_CAVER/'

for image_name in os.listdir(RGB_sal):
    if os.path.exists(save_path + image_name):
        # If the file exists, then run your code
        print(image_name, "111")
    else:
        print(image_name)
        gt = cv2.imread(RGB_sal + image_name)
        label = read_image_from_txt(label_txt + image_name.replace(".png", '.jpg') + '.txt')  # Region labels
        # Get the list of all classes in the image
        all_classes = np.unique(label)
        sal1 = cv2.imread(RGB_sal + image_name)
        sal2 = cv2.imread
