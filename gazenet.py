import keras
import os
import sys
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense,LSTM
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
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
            learning_rate,
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

        # build the model from the scratch
        input_img = Input((360, 360, 3))

        #block 1
        x = Conv2D(96, (11,11), strides=(4, 4), padding='valid')(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3,3))(x)

        #block 2
        x = Conv2D(256, (5,5), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2,2))(x)

        #block 3
        x = Conv2D(256, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2,2))(x)

        # fully-connected layers
        x = Flatten()(x)
        x = Dense(4096, activation = 'relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation = 'relu')(x)
        mlp = Dropout(0.5)(x)


        # LSTM
        LSTMLayer = LSTM(6,activation = 'tanh',recurrent_activation = 'hard_sigmoid')(mlp)
        output = Dense(6, activation='softmax')(LSTMLayer)
        model = Model(inputs = input_img,outputs = output)
        adam = optimizers.Adam(lr = self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer='adam')

    def train(self):
        # categorical_labels = to_categorical(int_labels, num_classes=None)

    def load_data(self):
        print("Hello world")

	def save_model_weights(self, folder_path, suffix):
		# Helper function to save your model / weights.
		self.model.save_weights(folder_path + 'weights-' +  str(suffix) + '.h5')
		self.model.save(folder_path + 'model-' +  str(suffix) + '.h5')

	def load_model(self, model_file):
		# Helper function to load an existing model.
		from keras.models import load_model
		self.model = load_model(model_file)

	def load_model_weights(self,weight_file):
		# Helper funciton to load model weights.
		self.model.load_weights(weight_file)


def main(args):
    learning_rate = 0.0001
    gazenet = GazeNet(learning_rate);
    # gazenet.load_data();
    gazenet.train();

if __name__ == '__main__':
	main(sys.argv)
