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
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras import backend as K
K.set_image_dim_ordering('tf')
import keras.callbacks
from keras import regularizers
from GazeNetModels import GazeNet_3D,GazeNet_2D,GazeNet_sequential_2D,GazeNet_native
import theano
import gazeWholeGenerator as gaze_gen_whole
import gazenetGenerator as gaze_gen
from compare_predict_truth import compare1

# global param
dataset_path = 'gaze_dataset_old/'
learning_rate = 0.00001
time_steps = 32
num_classes = 6
batch_size = 1
time_skip = 2
origin_image_size = 360    # size of the origin image before the cropWithGaze
img_size = 128    # size of the input image for network
num_channel = 3
steps_per_epoch=161
epochs=1
validation_step=20
total_num_epoch = 101
trainGenerator = gaze_gen.GazeDataGenerator(validation_split=0.2)
train_data = trainGenerator.flow_from_directory(dataset_path, subset='training',time_steps=time_steps,
                                                batch_size=batch_size, crop=False,
                                                gaussian_std=0.01, time_skip=time_skip)
val_data = trainGenerator.flow_from_directory(dataset_path, subset='validation', time_steps=time_steps,
                                              batch_size=batch_size, crop=False,
                                                gaussian_std=0.01, time_skip=time_skip)
gaze_net = GazeNet_sequential_2D(learning_rate,time_steps,num_classes,batch_size)
gaze_net.model.load_weights('model1/model_sequencial/20/weights.hdf5')
model = gaze_net.model

hist = model.fit_generator(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs, max_queue_size=1,
                validation_data=val_data, validation_steps=validation_step, callbacks=[reduce_lr],shuffle=False)


inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functor = K.function([inp]+ [K.learning_phase()], outputs ) # evaluation function

# Testing
test = np.random.random(input_shape)[np.newaxis,...]
layer_outs = functor([test, 1.])
print layer_outs



confusion_matrix = np.zeros((6,6))
for i in range(20):
    [img_seq,gaze_seq],label = next(train_data)
    y = model.predict([img_seq,gaze_seq],batch_size = 1)
    predicted_label = y.argmax(axis=1)
    grounded_label = label.argmax(axis=1)
    for j in range(len(predicted_label)):
        confusion_matrix[predicted_label[j],grounded_label[j]] = confusion_matrix[predicted_label[j],grounded_label[j]]+1
print(confusion_matrix)
