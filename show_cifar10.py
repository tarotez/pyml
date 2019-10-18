from PIL import Image
import numpy as np
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
imgs = []
for i in range(100):
	a = X_train[i,:,:,:]
	t = np.transpose(a,(1,2,0))
	imgs.append(Image.fromarray(np.uint8(t)))
