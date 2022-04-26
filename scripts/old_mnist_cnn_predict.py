"""Predicts using a simple convnet on the MNIST dataset.
"""


import numpy as np

np.random.seed(1337)  # for reproducibility

import os
import pickle

from keras.datasets import mnist
from keras.layers import (
    Activation,
    Convolution2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from keras.models import Sequential, model_from_json
from keras.utils import np_utils
from PIL import Image, ImageOps

with open("json_model_mnist_cnn.pkl", "rb") as f:
    json_string = pickle.load(f)

model = model_from_json(json_string)
model.load_weights("model_weights_mnist_cnn.h5")

# classification parameters
nb_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_test = X_test.astype("float32")
X_test /= 255

# convert class vectors to binary class matrices
Y_test = np_utils.to_categorical(y_test, nb_classes)

print("now using test set:")
i = 0
for sample in X_test:
    x_test = np.array([sample]).transpose((0, 2, 3, 1))
    y_est = model.predict(x_test, verbose=0)
    print("y_test:", Y_test[i])
    print("y_est:", y_est)
    print(" ")
    i += 1

print(" ")
print("now using our own images:")
input_dir = "number_images"
imagefiles = os.listdir(input_dir)
for filename in imagefiles:
    input_path = input_dir + "/" + filename
    colorImg = Image.open(input_path, "r")
    grayImg = ImageOps.invert(ImageOps.grayscale(colorImg))
    imgArray = np.asarray(grayImg) / 255
    x_test = np.array([np.array([imgArray])])
    y_est = model.predict(x_test, verbose=0)
    print("filename = ", filename)
    print("y_est = ", y_est)
    print(" ")
