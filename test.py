
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
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout,Conv3D,MaxPooling3D
from keras.layers import Activation, BatchNormalization, MaxPooling2D, Concatenate
import time,argparse
import math
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import Input
# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras import backend as K
K.set_image_dim_ordering('tf')
import keras.callbacks

import gazenetGenerator as gaze_gen
from compare_predict_truth import compare1
import gaze_net1 as gaze_net

dataset_path = 'gaze_dataset_old/'
learning_rate = 0.00007
time_steps = 32
num_classes = 6
batch_size = 1
time_skip = 2
origin_image_size = 360    # size of the origin image before the cropWithGaze
img_size = 128    # size of the input image for network
num_channel = 3
steps_per_epoch=28*6
epochs=5
validation_step=20
total_num_epoch = 101
x = [np.random.random(size = [1,32,256,256,3]),np.random.random(size = [1,32,3])]
y = np.random.randint(0,high = 7,size = (1,6))
gaze = gaze_net.GazeNet(learning_rate,time_steps,num_classes,batch_size)
model = gaze.model
while True:
    model.fit(x,y)
