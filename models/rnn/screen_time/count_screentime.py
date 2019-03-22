# -*- coding: utf-8 -*-

import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout
from skimage.transform import resize

count = 0
videoFile = "data/countData/Tom and Jerry 3.mp4"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5)  # frame rate
x = 1
while (cap.isOpened()):
    frameId = cap.get(1)  # current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename = "20news-bydate-test%d.jpg" % count;
        count += 1
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")

'''
After extracting the frames from the new video, we will now load the 20news-bydate-test.csv file which contains the names of each extracted frame
'''
test = pd.read_csv('data/20news-bydate-test.csv')

X = []  # creating an empty array
for img_name in test.Image_ID:
    img = plt.imread('data/countData' + img_name)
    X.append(img)  # storing each image in array X
X = np.array(X)  # converting list to array

y = test.Class
dummy_y = np_utils.to_categorical(y)  # one hot encoding Classes

'''
Next, we will import the images for testing and then reshape them as per the requirements of the aforementioned pretrained model:
'''
test_image = []
for img_name in test.Image_ID:
    img = plt.imread('data/countData' + img_name)
    test_image.append(img)
test_img = np.array(test_image)

test_image = []
for i in range(0, test_img.shape[0]):
    a = resize(test_img[i], preserve_range=True, output_shape=(224, 224)).astype(int)
    test_image.append(a)
test_image = np.array(test_image)

'''
We need to make changes to these images similar to the ones we did for the training images. 
We will preprocess the images, use the base_model.predict() 
function to extract features from these images using the VGG16 pretrained model, reshape these images to 1-D form, 
and make them zero-centered:
'''
# We will now load the VGG16 pretrained model and store it as base_model:
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(224, 224, 3))  # include_top=False to remove the top layer

# preprocessing the images
test_image = preprocess_input(test_image, mode='tf')

# extracting features from the images using pretrained model
test_image = base_model.predict(test_image)

# converting the images to 1-D form
test_image = test_image.reshape(186, 7*7*512)

# zero centered images
test_image = test_image/test_image.max()

#initialize a model
model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
model.add(Dense(3, activation='softmax'))    # output layer
# ii. Compiling the model
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# iii. Training the model
#model.fit(20news-bydate-train, y_train, epochs=100, validation_data=(X_valid, y_valid)

predictions = model.predict_classes(test_image)

print("The screen time of JERRY is", predictions[predictions==1].shape[0], "seconds")
print("The screen time of TOM is", predictions[predictions==2].shape[0], "seconds")