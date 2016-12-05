#! /bin/sh

# Abalone
wget https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names

# Iris
wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names

# AirQuality
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip
unzip AirQualityUCI.zip
rm AirQualityUCI.zip
rm AirQualityUCI.xlsx
