#!/bin/sh

mkdir -p data

# Abalone
wget -nc https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data -P data
wget -nc https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names -P data

# Iris
wget -nc https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -P data
wget -nc https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names -P data

# AirQuality
if ! [ -f data/AirQualityUCI.* ]; then
  wget https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip
  unzip AirQualityUCI.zip -d data
  rm AirQualityUCI.zip
  rm data/AirQualityUCI.xlsx
fi
