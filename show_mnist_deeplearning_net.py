# -*- coding: utf-8 -*-
import pickle
import sys
# from mnist import load_data
f = open('../data/mnist.pkl', 'rb')
train_set, valid_set, test_set = pickle.load(f)
train_set_x, train_set_y = train_set
i = 3
for y in range(0,28):
	for x in range(0,28):
	  if train_set_x[i][y*28+x]<0.5:
	    sys.stdout.write(" ")
	  elif train_set_x[i][y*28+x]<0.8:
	    sys.stdout.write("+")
	  else:
	    sys.stdout.write("*")
	sys.stdout.write("\n")
print("this is labeled ", train_set_y[i])
