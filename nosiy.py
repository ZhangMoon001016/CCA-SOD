import cv2
import os
import numpy as np
import time

def noise(img, SNR):
    h = img.shape[0]
    w = img.shape[1]
    img1 = img.copy()
    sp = h * w   # Calculate the number of pixels in the image
    NP = int(sp * (1 - SNR))   # Calculate the number of salt and pepper noise points in the image
    print(SNR, NP)
    for i in range(NP):
        randx = np.random.randint(1, h - 1)   # Generate a random integer between 1 and h-1
        randy = np.random.randint(1, w - 1)   # Generate a random integer between 1 and w-1
        if np.random.random() <= 0.5:   # np.random.random() generates a float between 0 and 1
            img1[randx, randy] = 0  # Set to black (salt noise)
        else:
            img1[randx, randy] = 255  # Set to white (pepper noise)
    return img1

image_path1 = 'E:/2/data/SIP/depth/'
SNR = ['0.90']

for snr in SNR:
    save_path1 = 'E:/2/data/SIP/ddd/' + snr + '/'
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
    for image_name in os.listdir(image_path1):
        # Read the image
        image = cv2.imread(image_path1 + image_name)
        noisy_img = noise(image, float(snr))
        # Save the image
        cv2.imwrite(save_path1 + image_name, noisy_img)
    print("Save done!")
