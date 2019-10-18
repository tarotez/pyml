from PIL import Image
import numpy as np
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
imgs = []
for i in range(100):
  a = X_train[i,:,:]
  imgs.append(Image.fromarray(np.uint8(a)))
