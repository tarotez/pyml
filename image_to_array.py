# copy image to an array (sampleNum, channelNum, img_rows, img_cols)
from __future__ import print_function
import numpy as np
import os
from PIL import Image
from PIL import ImageOps

def image2array(input_dir_prefix, nb_classes, img_tensor_shape):
	np.random.seed(1337)  # for reproducibility
	img_rows, img_cols, img_colors = img_tensor_shape
	sampleNum = 0
	for classid in list(range(nb_classes)):
		input_dir = input_dir_prefix + str(classid)
		imagefiles = os.listdir(input_dir)
		for filename in imagefiles:
			if not filename.startswith("."):
				sampleNum +=1
	print('sampleNum = ', sampleNum)			
	X = np.zeros((sampleNum, img_colors, img_rows, img_cols))
	y = np.zeros(sampleNum)

	sampleid = 0
	for classid in list(range(nb_classes)):
		input_dir = input_dir_prefix + str(classid)
		imagefiles = os.listdir(input_dir)
		for filename in imagefiles:
			if not filename.startswith("."):
				input_path = input_dir + "/" + filename	
				colorImg = np.asarray(Image.open(input_path,'r'))
				transposedImg = np.transpose(colorImg, (2,0,1))
				X[sampleid,:,:,:] = transposedImg
				y[sampleid] = classid
				sampleid += 1

	return (X, y)
