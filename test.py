import os
import sys
import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense,LSTM
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, TimeDistributed, Embedding
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, RepeatVector
from keras.layers import Activation, BatchNormalization, MaxPooling2D
import time
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import Input
from keras import backend as K
K.set_image_dim_ordering('tf')



data_dim = 16
num_classes = 10
batch_size = 32

class GazeNet():
    def __init__(self,
            learning_rate = 0.0001,
            num_classes = 6,

            timesteps = 4,
            num_classes = 6,
            batch_size = 32,
            data_dim = 4096,):
        self.learning_rate = learning_rate
        self.timesteps = timesteps
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.data_dim = data_dim
        self.model = self.create_mode()


    def create_mode(self):

        model = Sequential()

        adam = optimizers.Adam(lr = self.learning_rate)
        model = Model(inputs = main_input,outputs = main_input)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print(model.summary())

        return model


def main(args):
    gaze_net = GazeNet()
    # image_input = Input(shape=(224, 224, 3))
    # encoded_image = gaze_net(image_input)
    # SVG(model_to_dot(gaze_net.model).create(prog='dot', format='svg'))
    # plot_model(gaze_net.model, to_file='model.png')

if __name__ == '__main__':
	main(sys.argv)
