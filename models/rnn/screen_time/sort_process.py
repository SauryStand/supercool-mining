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

img = plt.imread('data/preProcessData/frame0.jpg')  # reading image using its name
plt.imshow(img)

data = pd.read_csv('data/mapping.csv')

print(data.head())

X = []  # creating an empty array
for img_name in data.Image_ID:
    img = plt.imread('data/preProcessData/' + img_name)
    X.append(img)  # storing each image in array X
X = np.array(X)  # converting list to array

y = data.Class
dummy_y = np_utils.to_categorical(y)  # one hot encoding Classes

image = []
for i in range(0, X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224, 224)).astype(int)  # reshaping to 224*224*3
    image.append(a)
X = np.array(image)

X = preprocess_input(X, mode='tf')  # preprocessing the input data

X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3,
                                                      random_state=42)  # preparing the validation set

# We will now load the VGG16 pretrained model and store it as base_model:
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(224, 224, 3))  # include_top=False to remove the top layer

'''
We will make predictions using this model for X_train and X_valid, get the features, and then use those features to retrain the model.
'''
X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape

'''
The shape of X_train and X_valid is (208, 7, 7, 512), (90, 7, 7, 512) respectively. 
In order to pass it to our neural network, we have to reshape it to 1-D.
'''
X_train = X_train.reshape(208, 7 * 7 * 512)  # converting to 1-D
X_valid = X_valid.reshape(90, 7 * 7 * 512)

train = X_train / X_train.max()  # centering the data
X_valid = X_valid / X_train.max()


'''
Finally, we will build our model. This step can be divided into 3 sub-steps:
Building the model
Compiling the model
Training the model
'''
model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
model.add(Dense(3, activation='softmax'))    # output layer

model.summary()

'''
We have a hidden layer with 1,024 neurons and an output layer with 3 neurons (since we have 3 classes to predict). Now we will compile our model:
'''
# ii. Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

'''
In the final step, we will fit the model and simultaneously also check its performance on the unseen images, i.e., validation images:
'''

'''
In the final step, we will fit the model and simultaneously also check its performance on the unseen images, i.e., validation images:
'''
# iii. Training the model
model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))



#next step was train the model
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
        filename = "data/countData/test%d.jpg" % count;
        count += 1
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")

test = pd.read_csv('data/test.csv')
#pre array
test_image = []
for img_name in test.Image_ID:
    img = plt.imread('data/countData/' + img_name)
    test_image.append(img)
test_img = np.array(test_image)

test_image = []
for i in range(0,test_img.shape[0]):
    a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
    test_image.append(a)
test_image = np.array(test_image)

'''
We need to make changes to these images similar to the ones we did for the training images. We will preprocess the images, use the base_model.predict() 
function to extract features from these images using the VGG16 pretrained model, reshape these images to 1-D form, and make them zero-centered:
'''

# preprocessing the images
test_image = preprocess_input(test_image, mode='tf')

# extracting features from the images using pretrained model
test_image = base_model.predict(test_image)

# converting the images to 1-D form
test_image = test_image.reshape(186, 7*7*512)

# zero centered images
test_image = test_image/test_image.max()

#use the model that we trained before
predictions = model.predict_classes(test_image)

print("The screen time of JERRY is", predictions[predictions==1].shape[0], "seconds")
print("The screen time of TOM is", predictions[predictions==2].shape[0], "seconds")