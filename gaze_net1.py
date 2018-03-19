# -*- coding: utf-8 -*-
"""gaze-net.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xiH7xOldq99KgnVDsWuT9GR9mnsU0ZXa
"""

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

import gazenetGenerator1 as gaze_gen

# global param
dataset_path = '../gaze-net/gaze_dataset'
learning_rate = 0.0001
time_steps = 32
num_classes = 6
batch_size = 4
time_skip = 2
origin_image_size = 360    # size of the origin image before the cropWithGaze
img_size = 128    # size of the input image for network
num_channel = 3
steps_per_epoch=5
epochs=1
validation_step=20
total_num_epoch = 40


class GazeNet():
    def __init__(self):
        self.learning_rate = 0.0001
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.model = self.create_model()

    def convolution(self, kernel_size = 5):
        def f(input):
            filters = 96
            conv1 = Conv2D(filters, kernel_size, strides=(3, 3), padding='valid', activation=None)(input)
            conv1 = Conv2D(filters, kernel_size, strides=(2, 2), padding='valid', activation=None)(conv1)
            conv1 = Conv2D(filters, kernel_size, strides=(2, 2), padding='valid', activation=None)(conv1)
            return conv1
        return f

    def lstm(self):
        def f(input):
            lstm = LSTM(128,return_sequences=True)(input)
            lstm = LSTM(96)(lstm)
            return lstm
        return f

    def create_model(self):
        image = Input(shape=(self.time_steps,256,256,3,))
        def input_reshape(image):
            return tf.reshape(image,[self.batch_size*self.time_steps,256,256,3])
        image_reshaped = Lambda(input_reshape)(image)
        image_embedding = self.convolution()(image_reshaped)
        flatten = Flatten()(image_embedding)

        gaze = Input(shape=(self.time_steps,3,))
        def input_gaze_reshape(input):
            return tf.reshape(input,[self.batch_size*self.time_steps,1,3])
        gaze_reshaped = Lambda(input_gaze_reshape)(gaze)
        print(gaze_reshaped)
        gaze_embedding = self.lstm()(gaze_reshaped)
        print(gaze_embedding)


        merged = Concatenate()([flatten, gaze_embedding])
        hidden = Dense(6)(merged)
        def classify(input):
            return tf.nn.softmax(input)
        def mean_value(input):
            res = tf.reshape(input,[self.batch_size,self.time_steps,num_classes])
            return tf.reduce_mean(res,1)
        hidden = Lambda(mean_value)(hidden)
        output = Lambda(classify)(hidden)

        model = Model(input=[image, gaze], output=output)


        adam = optimizers.Adam(lr = self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print(model.summary())
        return model
    def save_model_weights(self,save_path):
		# Helper function to save your model / weights.

        self.model.save_weights(save_path)
        # self.model.save(save_path)

        # return suffix
import keras.callbacks
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
from keras.callbacks import History


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)

	return parser.parse_args()

def main(args):
    # generate model
    args = parse_arguments()
    gaze_net = GazeNet()
    model = gaze_net.model
    #     plot_model(model, to_file='model.png')
    print("generate model!")
    if args.train == 1:
    # generatr generator
        for i in range(total_num_epoch):
            save_path = 'model1/'+str(i) + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            trainGenerator = gaze_gen.GazeDataGenerator(validation_split=0.2)
            train_data = trainGenerator.flow_from_directory(dataset_path, subset='training',time_steps=time_steps,
                                                            batch_size=batch_size, crop=False,
                                                            gaussian_std=0.01, time_skip=time_skip, crop_with_gaze=False
                                                          )
            val_data = trainGenerator.flow_from_directory(dataset_path, subset='validation', time_steps=time_steps,
                                                          batch_size=batch_size, crop=False,
                                                            gaussian_std=0.01, time_skip=time_skip, crop_with_gaze=False
                                                          )
            # [img_seq, gaze_seq], output = next(trainGeneratorgDirectory)
            print("fetch data!")
            # start training
            # checkpointsString = "models/" + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'

            # callbacks = gaze_net.save_model_weights(checkpointsString)
            # history = History()
            # checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)

            hist = model.fit_generator(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs,
                            validation_data=val_data, validation_steps=validation_step, shuffle=False)
            print("finished training!")
            print(hist.history)
            file = open(save_path + 'losses.txt','a')
            file.writelines(["%s\n" % loss  for loss in hist.history.values()])
            if i%10 == 0:
                model.save_weights( save_path + 'weights.hdf5')
    else:

        model.load_weights('model1/'+'weights.hdf5', by_name=False)
        testGenerator = gaze_gen.GazeDataGenerator(validation_split = 0.2)
        test_data = testGenerator.flow_from_directory(dataset_path, subset='training',time_steps=time_steps,
                                                        batch_size=batch_size, crop=False,
                                                        gaussian_std=0.01, time_skip=time_skip, crop_with_gaze=True,
                                                        crop_with_gaze_size=128)
        predicted_labels = model.predict_generator(test_data,steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
        print(predicted_labels)
if __name__ == '__main__':
    main(sys.argv)
