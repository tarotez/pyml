from PIL import Image
import numpy as np
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
for i in range(10):
	a = X_train[i,:,:]
	img = Image.fromarray(np.uint8(a))
	img.show()
