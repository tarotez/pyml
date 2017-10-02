'''Predicts using a simple convnet on the MNIST dataset.
'''

from __future__ import print_function
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
import os
import pickle
from matplotlib import pyplot as plt
from image_to_array import image2array

with open('json_model_face_cnn.pkl','rb') as f:
	json_string = pickle.load(f)

model = model_from_json(json_string)
model.load_weights('model_weights_face_cnn.h5')

input_dir_prefix = 'faces/class_'
nb_classes = 6
nb_filters = 32
img_rows, img_cols, img_colors = 32, 32, 3
img_tensor_shape = (img_rows, img_cols, img_colors)
(X, y_orig) = image2array(input_dir_prefix, nb_classes, img_tensor_shape)

# visualization
def draw_digit(data, row, col, n):
    plt.subplot(row, col, n)    
    plt.imshow(data)
    plt.gray()

X = X.astype('float32')
X /= 255
X_train = np.random
show_size = 5

get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[4].output])

layers = get_3rd_layer_output([X_train[0:show_size]])[0]

show_size = 10
plt.figure(figsize=(20,20))

for img_index, filters in enumerate(layers, start=1):
    for filter_index, mat in enumerate(filters):
        pos = (filter_index)*10+img_index
        draw_digit(mat, nb_filters, show_size, pos)
plt.show()
