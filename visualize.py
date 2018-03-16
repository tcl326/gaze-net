import os
import sys
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Lambda
from keras.engine.topology import Input
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers import Activation, BatchNormalization, MaxPooling2D
import time
import math
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import Input
# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('tf')

import gazenetGenerator as gaze_gen

dataset_path = 'gaze_dataset'
test_path = 'test/'
bs = 4
ts = 32
target_size = (360,640)
img_size = 128

if not os.path.exists(test_path):
	os.makedirs(test_path)

trainGenerator = gaze_gen.GazeDataGenerator(validation_split=0.2)

# visualize for without crop with gaze
# train_data = trainGenerator.flow_from_directory(dataset_path, subset='training',time_steps=ts, batch_size=bs,
#                                                 target_size= target_size, crop=False, gaussian_std=3)
#
# [img_seq, gaze_seq], output = next(train_data)
#
# print(img_seq.shape)
# print(gaze_seq.shape)
# print(gaze_seq)
# for i in range(ts):
# 	img = img_seq[0][i]
# 	gaze = gaze_seq[0][i]
# 	print(output[0])
# 	x = gaze[1]
# 	y = gaze[2]
# 	left = int(max(0, x-10))
# 	right = int(min(x+10, target_size[1]-1))
# 	above = int(max(0, y-10))
# 	bottom = int(min(y+10, target_size[0]-1))
# 	img = img/255.0
# 	img[above:bottom, left:right, 0] = 0
# 	img[above:bottom, left:right, 1] = 	0
# 	img[above:bottom, left:right, 2] = 	1
# 	# print(img)
# 	img_name = test_path + str("%03d" % i) + '.png'
# 	plt.imsave(img_name, img)

# visualize for crop with gaze
train_data = trainGenerator.flow_from_directory(dataset_path, subset='training',time_steps=ts,
                                                batch_size=bs, crop=False, target_size=target_size,
                                                gaussian_std=0.01, time_skip=3, crop_with_gaze=True,
                                               crop_with_gaze_size=img_size)
img_seq, output = next(train_data)
print(output[0])
for i in range(ts):
	img = img_seq[0][i]/255.0
	img_name = test_path + str("%03d" % i) + '.png'
	plt.imsave(img_name, img)
