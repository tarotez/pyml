'''Predicts using a simple convnet on the MNIST dataset.
'''
from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
import os
import pickle
from PIL import Image
from PIL import ImageOps

input_path = 'number.png'

with open('json_model_mnist_cnn.pkl','rb') as f:
     json_string = pickle.load(f)

model = model_from_json(json_string)
model.load_weights('model_weights_mnist_cnn.h5')

colorImg = Image.open(input_path,'r')
grayImg = ImageOps.grayscale(colorImg)
# grayImg = ImageOps.invert(grayImg)
a = np.asarray(grayImg)
rows, cols = a.shape
x_test = np.zeros((1,1,rows,cols))
x_test[0,0,:,:] = a
x_test = x_test.transpose([0,2,3,1])

y_est = model.predict(x_test, verbose=0)
print('y_est is ', y_est)
est_max_idx = np.argmax(y_est)
print('this number is ', est_max_idx)
