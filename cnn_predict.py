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

def predict(X_test, Y_test):
	
	# np.random.seed(1337)  # for reproducibility

	with open('json_model_cnn.pkl','rb') as f:
		json_string = pickle.load(f)

	model = model_from_json(json_string)
	model.load_weights('model_weights_cnn.h5')

	print('now using test set:')
	i = 0
	correct = 0
	for sample in X_test:
		x_test = np.array([sample])
		y_est = model.predict(x_test, verbose=0)
		est_max_idx = np.argmax(y_est)
		if Y_test[i][est_max_idx]:
			correct += 1
		print('y_test:', Y_test[i])
		print('y_est:', y_est)	   
		print(' ')
		i += 1
	print('test precision = ', correct / i)
	print(' ')

