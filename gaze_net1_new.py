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
from GazeNetModels_1 import GazeNet_3D,GazeNet_2D,GazeNet_sequential_2D

import gazeWholeGenerator as gaze_gen
# import gazeWholeGenerator as gaze_gen
from compare_predict_truth import compare1

# global param
dataset_path = 'gaze_dataset/'
learning_rate = 0.00001
time_steps = 128
num_classes = 5
batch_size = 1
time_skip = 2
origin_image_size = 360    # size of the origin image before the cropWithGaze
img_size = 128    # size of the input image for network
num_channel = 3
steps_per_epoch=161
epochs=1
validation_step=20
total_num_epoch = 101


import keras.callbacks
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
from keras.callbacks import History

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model',type=str, default='')
    parser.add_argument('--model_type',dest='model_type',type=str,default='')
    return parser.parse_args()

def train(model,args):

    pre_trained_model = args.model

    if pre_trained_model != '':
        model.load_weights(pre_trained_model)
    trainGenerator = gaze_gen.GazeDataGenerator(validation_split=0)
    train_data = trainGenerator.flow_from_directory(dataset_path, subset='training',time_steps=time_steps,
                                                    batch_size=batch_size, crop=False,
                                                    gaussian_std=0.01, time_skip=time_skip)
    # val_data = trainGenerator.flow_from_directory(dataset_path, subset='validation', time_steps=time_steps,
    #                                               batch_size=batch_size, crop=False,
    #                                                 gaussian_std=0.01, time_skip=time_skip)
    for i in range(total_num_epoch):
        print(i)
        save_path = 'model1/model_sequencial/'+str(i) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.000001)
        hist = model.fit_generator(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs, max_queue_size=1,
                        validation_data=train_data, validation_steps=validation_step, callbacks=[reduce_lr],shuffle=False)
        file = open(save_path + 'losses.txt','a')
        file.writelines(["%s\n" % loss  for loss in hist.history.values()])
        if i%10 == 0:
            model.save_weights( save_path + 'weights.hdf5')
def test(model,pre_trained_model):
    if pre_trained_model!='':
        model.load_weights(pre_trained_model, by_name=False)
    testGenerator = gaze_gen.GazeDataGenerator()
    test_data = testGenerator.flow_from_directory(dataset_path, subset='training',time_steps=time_steps,
                                                    batch_size=batch_size, crop=False,
                                                    gaussian_std=0.01, time_skip=time_skip, crop_with_gaze=False,
                                                    crop_with_gaze_size=128)
    err_list = compare1(model, test_data, 49)
def main(args):
    args = parse_arguments()
    gaze_net = GazeNet_sequential_2D(learning_rate,time_steps,num_classes,batch_size)
    model = gaze_net.model
    print("generate model!")
    if args.train == 1:
        train(model,args)
    else:
        test(model,args.model)

if __name__ == '__main__':
    main(sys.argv)
