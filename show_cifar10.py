from PIL import Image
import numpy as np
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
imgs = []
i = 5
a = X_train[i,:,:,:]
img = Image.fromarray(np.uint8(a))
