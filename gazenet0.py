import os
import sys
import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense,LSTM
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, RepeatVector
from keras.layers import Activation, BatchNormalization, MaxPooling2D
import time
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import Input
from keras import backend as K
K.set_image_dim_ordering('tf')

class GazeNet():
    def __init__(self,
            learning_rate = 0.0001,
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
        # build the model from the scratch
        # input_img = Input(shape = (128,128,3))

        #block 1
        # x = Conv2D(96, (11,11), strides=(4, 4), padding='valid')(input_img)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = MaxPooling2D((3,3))(x)
        model.add(Conv2D(96,(11,11),strides = (4,4),
                            padding = 'valid',
                            activation = 'relu',
                            input_shape = (128,128,3)
                            ))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (3,3)))

        #block 2
        # x = Conv2D(256, (5,5), padding='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = MaxPooling2D((2,2))(x)
        # model.add(Conv2D(256,(5,5),padding = 'same'))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(2,2))
        #block 3
        # x = Conv2D(256, (3,3), padding='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        model.add(Conv2D(256,(3,3),padding = 'same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # x = Conv2D(256, (3,3), padding='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        model.add(Conv2D(256,(3,3),padding = 'same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))


        # x = Conv2D(256, (3,3), padding='same')(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = MaxPooling2D((2,2))(x)
        model.add(Conv2D(256,(3,3),padding = 'same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2,2))


        # fully-connected layers
        # x = Flatten()(x)
        # x = Dense(4096, activation = 'relu')(x)
        # x = Dropout(0.5)(x)
        # x = Dense(4096, activation = 'relu')(x)
        # mlp = Dropout(0.5)(x)
        model.add(Flatten())
        model.add(Dense(4096,activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096,activation = 'relu'))
        model.add(Dropout(0.5))

        # model.add(TimeDistributed(Dense(4096)))
        # print()
        # def mean_value(input):
            # return [np.mean(input[i:i+timestep]) for i in range((len(input)/timestep))]
        # model.add(Reshape())
        lstmModel = Sequential()
        lstmModel.add(TimeDistributed(model,input_shape = (32,4096)))
        lstmModel.add(LSTM(6))
        adam = optimizers.Adam(lr = self.learning_rate)
        # model.compile(loss='categorical_crossentropy', optimizer='adam')
        lstmModel.compile(loss = 'categorical_crossentropy',optimizer = 'adam')
        # print(model.summary())

        print(lstmModel.summary())

        return model


def main(args):
    gaze_net = GazeNet()
    image_input = Input(shape=(224, 224, 3))
    # encoded_image = gaze_net(image_input)
    # SVG(model_to_dot(gaze_net.model).create(prog='dot', format='svg'))
    # plot_model(gaze_net.model, to_file='model.png')

if __name__ == '__main__':
	main(sys.argv)
