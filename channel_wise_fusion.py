import cv2
import matplotlib.pyplot as plt
import os

image_path = 'E:/2/data/COME-E/RGB/'
depth_path = 'E:/2/data/COME-E/depth/'
save_path1 = 'E:/2/data/COME-E/ddb/ddb1/'
save_path2 = 'E:/2/data/COME-E/dgd/dgd1/'
save_path3 = 'E:/2/data/COME-E/rdd/rdd1/'

if not os.path.exists(save_path1):
    os.makedirs(save_path1)
if not os.path.exists(save_path2):
    os.makedirs(save_path2)
if not os.path.exists(save_path3):
    os.makedirs(save_path3)

for image_name in os.listdir(image_path):
    image = cv2.imread(image_path + image_name)
    depth = cv2.imread(depth_path + image_name)

    # Resize depth image to match the size of the color image
    depth_resized = cv2.resize(depth, (image.shape[1], image.shape[0]))

    b, g, r = cv2.split(image)  # The order is b, g, r, not r, g, b
    d, _, _ = cv2.split(depth_resized)

    rdd = cv2.merge([d, d, r])  # b, g, r
    ddb = cv2.merge([b, d, d])
    dgd = cv2.merge([d, g, d])
    # rgd = cv2.merge([d, g, r])
    cv2.imwrite(save_path1 + image_name, ddb)
    cv2.imwrite(save_path2 + image_name, dgd)
    cv2.imwrite(save_path3 + image_name, rdd)

# image = cv2.merge([r, g, b])


# plt.figure(figsize=(10, 5))
# plt.subplot(2, 3, 1), plt.title('image')
# plt.imshow(image), plt.axis('off')
# plt.subplot(2, 3, 2), plt.title('image_gray')
# plt.imshow(image_gray, cmap='gray'), plt.axis('off')
# plt.subplot(2, 3, 3), plt.title('image_merged')
# plt.imshow(image_merged), plt.axis('off')
# plt.subplot(2, 3, 4), plt.title('r')
# plt.imshow(r, cmap='gray'), plt.axis('off')
# plt.subplot(2, 3, 5), plt.title('g')
# plt.imshow(g, cmap='gray'), plt.axis('off')
# plt.subplot(2, 3, 6), plt.title('b')
# plt.imshow(b, cmap='gray'), plt.axis('off')
# plt.show()
