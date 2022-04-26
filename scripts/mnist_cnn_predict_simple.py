import os
import pickle

import numpy as np
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

input_path = os.path.join(os.path.dirname(__file__), "..", "data", "number.png")

with open("json_model_mnist_cnn.pkl", "rb") as f:
    json_string = pickle.load(f)

model = model_from_json(json_string)
model.load_weights("model_weights_mnist_cnn.h5")

colorImg = Image.open(input_path, "r")
grayImg = ImageOps.grayscale(colorImg)
grayImg = ImageOps.invert(grayImg)
a = np.asarray(grayImg)
rows, cols = a.shape
x_test = np.zeros((1, 1, rows, cols))
x_test[0, 0, :, :] = a

y_est = model.predict(x_test, verbose=0)
print("y_est is ", y_est)
est_max_idx = np.argmax(y_est)
print("this number is ", est_max_idx)
