
import numpy as np
from image_to_array import image2array
import math
from keras.utils import np_utils
from cnn_train import train
from cnn_predict import predict
# np.random.seed(1337)  # for reproducibility

# the data, shuffled and split between train and test sets
input_dir_prefix = 'faces/class_'
nb_classes = 6
img_rows, img_cols, img_colors = 32, 32, 3
img_tensor_shape = (img_rows, img_cols, img_colors)
trainSampleRatio = 9/10
# batch_size = 3
batch_size = 32
# nb_epoch = 200
nb_epoch = 1
data_augmentation = True
# data_augmentation = False

(X, y_orig) = image2array(input_dir_prefix, nb_classes, img_tensor_shape)

# setup X
X = X.astype('float32')
X /= 255

# convert class vectors to binary class matrices
Y = np_utils.to_categorical(y_orig, nb_classes)
sampleNum = X.shape[0]
trainSampleNum = math.floor(sampleNum * trainSampleRatio)
print('sampleNum:', sampleNum)
randomIndices = np.random.permutation(list(range(sampleNum)))
X = X[randomIndices]
Y = Y[randomIndices]
print('X.shape: ', X.shape)
X_train = X[:trainSampleNum]
X_test = X[trainSampleNum+1:sampleNum]
Y_train = Y[:trainSampleNum]
Y_test = Y[trainSampleNum+1:]

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print(X_test.shape[0], 'test samples')
print('Y_test shape:', Y_test.shape)

train(X_train, Y_train, X_train, Y_train, img_colors, img_rows, img_cols, batch_size, nb_epoch, data_augmentation)
predict(X_test, Y_test)

