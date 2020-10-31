import os
import cv2
import numpy as np
import random

height = 360
width = 480



dir_path = 'D:\\major_proj\\resized_data'

for i in os.listdir(dir_path):
    quality = random.choice([10,20,30])
    img_name = i.split('.')[0]
    img = cv2.imread(os.path.join(dir_path,i))
    cv2.imwrite(f'D:\\major_proj\\compressed\\{img_name}.jpeg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


'''
=> Download dataset

dir_path = 'D:\\major\\coco\\train2017\\train2017'

temp = os.path.join(dir_path)

count = 0
for i in os.listdir(temp):
    temp_img = cv2.imread(os.path.join(dir_path,i))
    height,width,channels = temp_img.shape

    if width >= 480 and height >= 360:
        cv2.imwrite(f'D:\\major_proj\\data\\{i}',temp_img)
        count += 1


print(count)
'''


'''

=> model architecture

    img_height = 360
    img_width = 480
    img_channels = 3

    input_shape = (img_height, img_width, img_channels)
    img_input = k.Input(shape=input_shape)
    conv1 = layers.Conv2D(64,9, padding = 'same')(img_input)
    conv2 = layers.Conv2D(16,1, padding = 'same')(conv1)
    conv3 = layers.Conv2D(32,7, padding = 'same')(conv2)
    conv4 = layers.Conv2D(16,1, padding = 'same')(conv3)
    conv5 = layers.Conv2D(16,3, padding = 'same')(conv4)
    conv6 = layers.Conv2D(16,1, padding = 'same')(conv5)
    conv7 = layers.Conv2D(1,5, padding = 'same')(conv6)
    
    model = models.Model(img_input, conv7)

    return model

'''