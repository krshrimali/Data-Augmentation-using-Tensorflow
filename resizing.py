'''
Usage: python3 resizing.py 1.jpg 2.jpg 3.jpg

where:

1.jpg, 2.jpg, 3.jpg : are the input images
[clone the github repo to execute in the same way with the same images]

Reference: https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9

Reference Credits: Krutika Bapat (github.com/krutikabapat)

Images credits: Helen Dataset, OpenSource
'''

import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt

IMAGE_SIZE = 224

# referenced
def tf_resize_images(X_img_file_paths):
    X_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, (None, None, 3))
    tf_img = tf.image.resize_images(X, [IMAGE_SIZE,IMAGE_SIZE], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Each image is resized individually as different image may be of different size.
        for index, file_path in enumerate(X_img_file_paths):
            img = mpimg.imread(file_path)[:, :, :3] # Do not read alpha channel.
            resized_img = sess.run(tf_img, feed_dict = {X: img})
            X_data.append(resized_img)
    
    X_data = np.array(X_data, dtype = np.float32) # Convert to numpy
    return X_data

# create image paths list
image_paths = []

# append from command line arguments
for i in range(1, len(sys.argv)):
    image_paths.append(sys.argv[i])
print(image_paths)

# store returned list of numpy arrays
data = tf_resize_images(image_paths)

# iterate through the array, and save the image to the directory
# data is array of numpy arrays
# so data[i] used to write, not data
for i in range(len(data)):
    img = cv2.cvtColor(data[i], cv2.COLOR_RGB2BGR)
    cv2.imwrite("image_output" + str(i) + ".jpg", img)
    # unable to show the image output properly - to-do
# data = np.array(data)

