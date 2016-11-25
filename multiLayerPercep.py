import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import numpy as np
from sklearn import preprocessing
np.random.seed(2)

nb_epoch = 100
nb_internalnodes = 5

attrLabels = ["sepal length", "sepal width", "petal length", "petal width"]
categoryLabels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
nb_classes = len(categoryLabels)

le = preprocessing.LabelEncoder()
le.fit(categoryLabels);

datalist = []
depVar = []
f = open('iris.data', 'r')
for line in f:
    line = line.rstrip()
    elems = line.split(',')
    if len(elems) > 1:
        label = elems[-1]
        datalist.append(elems[:-1])
        depVar.append(le.transform([label])[0])
f.close()

X = np.float_(np.array(datalist))
y = np.array(depVar)
y_oneHot = np_utils.to_categorical(y, nb_classes)

sampleNum, attrNum = X.shape

model = Sequential()

model.add(Dense(nb_internalnodes, input_shape=(attrNum,)))
model.add(Dense(nb_classes, input_shape=(nb_internalnodes,)))
model.add(Activation("softmax"))

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy'])

model.fit(X, y_oneHot, nb_epoch = nb_epoch, verbose=1)
