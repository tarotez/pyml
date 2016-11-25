import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from sklearn import preprocessing

np.random.seed(2)

attrLabels = ["sepal length", "sepal width", "petal length", "petal width"]
categoryLabels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

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
        if label == categoryLabels[0] or label == categoryLabels[1]:
            datalist.append(elems[:-1])
            depVar.append(le.transform([label])[0])
f.close()

X = np.float_(np.array(datalist))
y = np.array(depVar)

sampleNum, attrNum = X.shape

model = Sequential()
model.add(Dense(1, input_shape=(attrNum,)))
model.add(Activation("sigmoid"))

model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=['accuracy'])

model.fit(X, y, verbose=1)
