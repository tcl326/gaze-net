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
import theano
from keras import regularizers

import gazeWholeGenerator as gaze_gen_whole
import gazenetGenerator as gaze_gen
from compare_predict_truth import compare1


# global param
dataset_path = 'gaze_dataset/'
learning_rate = 0.00001
time_steps = 32
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

class GazeNet_3D():
    def __init__(self,learning_rate,time_steps,num_classes,batch_size):
        self.learning_rate = 0.0001
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.model = self.create_model()

    def convolution(self, kernel_size = 3):
        def f(input):
            filters = 96
            conv1 = Conv2D(filters, kernel_size, strides=(3, 3), padding='valid', activation=None)(input)
            conv1 = BatchNormalization()(conv1)
            conv1 = MaxPooling2D(pool_size = 2, padding = 'valid')(conv1)
            conv1 = Dropout(0.5)(conv1)
            conv1 = Conv2D(filters, kernel_size, strides=(2, 2), padding='valid', activation=None)(conv1)
            conv1 = BatchNormalization()(conv1)
            conv1 = MaxPooling2D(pool_size = 2, padding = 'valid')(conv1)
            conv1 = Dropout(0.5)(conv1)
            conv1 = Conv2D(filters, kernel_size, strides=(2, 2), padding='valid', activation=None)(conv1)
            conv1 = BatchNormalization()(conv1)
            conv1 = MaxPooling2D(pool_size = 2, padding = 'valid')(conv1)
            conv1 = Dropout(0.5)(conv1)
            return conv1
        return f
    def convolution3D(self, kernel_size = 3):
        def f(input):
            filters = 48
            conv1 = Conv3D(filters, kernel_size, strides=(3, 3, 3), padding='valid', activation=None)(input)
            conv1 = BatchNormalization()(conv1)
            conv1 = MaxPooling3D(pool_size = (2,3,3), padding = 'valid')(conv1)
            conv1 = Dropout(0.5)(conv1)
            conv1 = Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', activation=None)(conv1)
            conv1 = BatchNormalization()(conv1)
            conv1 = MaxPooling3D(pool_size = (1,3,3),padding = 'valid')(conv1)
            conv1 = Dropout(0.5)(conv1)
            return conv1
        return f

    def lstm(self):
        def f(input):
            lstm = LSTM(128,return_sequences=True)(input)
            lstm = Dropout(0.5)(lstm)
            lstm = LSTM(128)(lstm)
            lstm = Dropout(0.5)(lstm)
            return lstm
        return f
    def lstm2D(self):
        def f(input):
            lstm = LSTM(128,return_sequences=True)(input)
            lstm = Dropout(0.5)(lstm)
            lstm = LSTM(128,return_sequences=False)(lstm)
            # lstm = LSTM(128,return_sequences=True)(lstm)
            lstm = Dropout(0.5)(lstm)
            return lstm
        return f
    def create_model(self):
        image = Input(shape=(self.time_steps,256,256,3,))
        def input_reshape(image):
            return tf.reshape(image,[-1,time_steps,256,256,3])
            # return tf.reshape(image,[-1,256,256,3])
        image_reshaped = Lambda(input_reshape)(image)
        # print(K.ndim(image_reshaped))
        # image_embedding = self.convolution()(image_reshaped)
        image_embedding = self.convolution3D()(image_reshaped)
        flatten = Flatten()(image_embedding)
        # flatten = K.
        print(K.ndim(flatten))


        gaze = Input(shape=(self.time_steps,3,))
        def input_gaze_reshape(input):
            return tf.reshape(input,[-1,self.time_steps,3])
        gaze_reshaped = Lambda(input_gaze_reshape)(gaze)
        gaze_embedding = self.lstm2D()(gaze_reshaped)
        def mean_value(input):
            return tf.reduce_mean(input,1)
        # gaze_embedding = Lambda(mean_value)(gaze_embedding)
        # print(gaze_embedding)


        merged = Concatenate()([flatten, gaze_embedding])
        # print(merged)
        hidden = Dense(128,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(merged)
        hidden = Dropout(0.5)(hidden)
        hidden = Dense(64,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(hidden)
        hidden = Dropout(0.5)(hidden)
        hidden = Dense(6,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(hidden)
        # print(hidden)
        def classify(input):
            return tf.nn.softmax(input)
        def mean_value(input):
            res = tf.reshape(input,[self.batch_size,self.time_steps,num_classes])
            return tf.reduce_mean(res,1)
        # hidden = Lambda(mean_value)(hidden)
        output = Lambda(classify)(hidden)

        model = Model(input=[image, gaze], output=output)


        adam = optimizers.Adam(lr = self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['mae', 'acc'])
        print(model.summary())
        # plot_model(model,to_file='model_3D.png',show_shapes=True, show_layer_names=True)
        return model
    def save_model_weights(self,save_path):
		# Helper function to save your model / weights.

        self.model.save_weights(save_path)
        # self.model.save(save_path)

        # return suffix

class GazeNet_2D():
    def __init__(self,learning_rate,time_steps,num_classes,batch_size):
        self.learning_rate = 0.0001
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.model = self.create_model()

    def convolution(self, kernel_size = 3):
        def f(input):
            filters = 96
            conv1 = Conv2D(filters, kernel_size, strides=(3, 3), padding='valid', activation=None)(input)
            conv1 = BatchNormalization()(conv1)
            conv1 = MaxPooling2D(pool_size = 2, padding = 'valid')(conv1)
            conv1 = Dropout(0.5)(conv1)
            conv1 = Conv2D(filters, kernel_size, strides=(2, 2), padding='valid', activation=None)(conv1)
            conv1 = BatchNormalization()(conv1)
            conv1 = MaxPooling2D(pool_size = 2, padding = 'valid')(conv1)
            conv1 = Dropout(0.5)(conv1)
            conv1 = Conv2D(filters, kernel_size, strides=(2, 2), padding='valid', activation=None)(conv1)
            conv1 = BatchNormalization()(conv1)
            conv1 = MaxPooling2D(pool_size = 2, padding = 'valid')(conv1)
            conv1 = Dropout(0.5)(conv1)
            return conv1
        return f

    def lstm(self):
        def f(input):
            lstm = LSTM(128,return_sequences=True)(input)
            lstm = Dropout(0.5)(lstm)
            lstm = LSTM(128)(lstm)
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
        def reshape_merge(input):
            return tf.reshape(input,[-1,self.time_steps,512])
        merged = Lambda(reshape_merge)(merged)
        merged = Flatten()(merged)
        def classify(input):
            return tf.nn.softmax(input)
        def mean_value(input):
            res = tf.reshape(input,[self.batch_size,self.time_steps,num_classes])
            return tf.reduce_mean(res,1)
        hidden = Dense(4096,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(merged)
        hidden = Dropout(0.5)(hidden)
        hidden = Dense(4096,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(hidden)
        hidden = Dropout(0.5)(hidden)
        hidden = Dense(6,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(hidden)
        output = Lambda(classify)(hidden)

        model = Model(input=[image, gaze], output=output)


        adam = optimizers.Adam(lr = self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['mae', 'acc'])
        print(model.summary())
        # plot_model(model,to_file='model_2D.png',show_shapes=True, show_layer_names=True)

        return model
    def save_model_weights(self,save_path):
		# Helper function to save your model / weights.

        self.model.save_weights(save_path)
        # self.model.save(save_path)

class GazeNet_sequential_2D():
    def __init__(self,learning_rate,time_steps,num_classes,batch_size):
        self.learning_rate = 0.0001
        self.batch_size = batch_size
        self.time_steps = time_steps
        print("hello world!!!!!!!!!!!!!!!")

        self.model = self.create_model()


    def convolution(self, kernel_size = 3):
        def f(input):
            filters = 96
            conv1 = Conv2D(filters, kernel_size, strides=(3, 3), padding='valid', activation=None)(input)
            conv1 = BatchNormalization()(conv1)
            conv1 = MaxPooling2D(pool_size = 2, padding = 'valid')(conv1)
            # conv1 = Dropout(0.5)(conv1)
            conv1 = Conv2D(filters, kernel_size, strides=(2, 2), padding='valid', activation=None)(conv1)
            conv1 = BatchNormalization()(conv1)
            conv1 = MaxPooling2D(pool_size = 2, padding = 'valid')(conv1)
            # conv1 = Dropout(0.5)(conv1)
            conv1 = Conv2D(filters, kernel_size, strides=(2, 2), padding='valid', activation=None)(conv1)
            conv1 = BatchNormalization()(conv1)
            conv1 = MaxPooling2D(pool_size = 2, padding = 'valid')(conv1)
            # conv1 = Dropout(0.5)(conv1)
            return conv1
        return f

    def lstm(self):
        def f(input):
            lstm = LSTM(128,return_sequences=True,stateful=True)(input)
            # lstm = Dropout(0.5)(lstm)
            lstm = LSTM(128,return_sequences=True,stateful=True)(lstm)
            return lstm
        return f

    def create_model(self):
        print("hello world!!!!!!!!!!!!!!!")
        image =Input(shape=(256,256,3,))
        def input_reshape(image):
            return tf.reshape(image,[-1,256,256,3])
        image_reshaped = Lambda(input_reshape)(image)
        image_embedding = self.convolution()(image_reshaped)
        flatten = Flatten()(image_embedding)
        def flatten_reshape(input):
            return tf.reshape(input,[self.batch_size,-1,384])
        flatten = Lambda(flatten_reshape)(flatten)
        # convout1_f = theano.function([model.get_input(train=False)], image_embedding.get_output(train=False))

        gaze = Input(shape=(3,))
        def input_gaze_reshape(input):
            return tf.reshape(input,[self.batch_size,-1,3])
        gaze_reshaped = Lambda(input_gaze_reshape)(gaze)
        # print(gaze_reshaped)
        gaze_embedding = self.lstm()(gaze_reshaped)
        # # print(gaze_embedding)
        # def lstm_gaze_reshaped(input):
        #     return tf.reshape(input,[self.time_steps,128])
        # gaze_embedding=Lambda(lstm_gaze_reshaped)(gaze_embedding)
        merged = Concatenate()([flatten, gaze_embedding])
        # merged = K.squeeze(merged,axis=0)
        # hidden = Dense(6)(merged)
        def reshape_merge(input):
            return tf.reshape(input,[-1,512])
        merged = Lambda(reshape_merge)(merged)
        def classify(input):
            return tf.nn.softmax(input)
        # def mean_value(input):
        #     res = tf.reshape(input,[self.batch_size,self.time_steps,num_classes])
        #     return tf.reduce_mean(res,1)
        hidden = Dense(512,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(merged)
        # hidden = Dropout(0.5)(hidden)
        hidden = Dense(512,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(hidden)
        # hidden = Dropout(0.5)(hidden)
        hidden = Dense(6,kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01))(hidden)
        # hidden = Lambda(mean_value)(hidden)
        # print(hidden)
        output = Lambda(classify)(hidden)
        def reshape_output(input):
            return tf.reshape(input,[1,-1,6])
        # output = Lambda(reshape_output)(output)
        model = Model(input=[image,gaze], output=output)

        adam = optimizers.Adam(lr = self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['mae', 'acc'])
        print(model.summary())
        # plot_model(model,to_file='model_sequencial_2D.png',show_shapes=True, show_layer_names=True)
        return model
    def save_model_weights(self,save_path):
		# Helper function to save your model / weights.
        self.model.save_weights(save_path)
        # self.model.save(save_path)

class GazeNet_native():
    def __init__(self,learning_rate,time_steps,num_classes,batch_size):
        self.learning_rate = 0.0001
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.model = self.create_model()

    def convolution(self, kernel_size = 3):
        def f(input):
            filters = 96
            conv1 = Conv2D(filters, kernel_size, strides=(3, 3), padding='valid', activation=None)(input)
            conv1 = BatchNormalization()(conv1)
            conv1 = MaxPooling2D(pool_size = 2, padding = 'valid')(conv1)
            conv1 = Dropout(0.5)(conv1)
            conv1 = Conv2D(filters, kernel_size, strides=(2, 2), padding='valid', activation=None)(conv1)
            conv1 = Dropout(0.5)(conv1)
            conv1 = Conv2D(filters, kernel_size, strides=(2, 2), padding='valid', activation=None)(conv1)
            conv1 = MaxPooling2D(pool_size = 2,padding = 'valid')(conv1)

            return conv1
        return f

    def lstm(self):
        def f(input):
            lstm = LSTM(128,return_sequences=True)(input)
            lstm = Dropout(0.5)(lstm)
            lstm = LSTM(128)(lstm)
            return lstm
        return f

    def create_model(self):
        print("!!!!!!!!!!!!!!!!!!!!!!!!")
        image = Input(shape=(self.time_steps,256,256,3,))
        def input_reshape(image):
            print("input reshape")
            return tf.reshape(image,[-1,256,256,3])
        image_reshaped = Lambda(input_reshape)(image)
        image_embedding = self.convolution()(image_reshaped)
        flatten = Flatten()(image_embedding)

        gaze = Input(shape=(self.time_steps,3,))
        def input_gaze_reshape(input):
            return tf.reshape(input,[-1,1,3])
        gaze_reshaped = Lambda(input_gaze_reshape)(gaze)
        print(gaze_reshaped)
        gaze_embedding = self.lstm()(gaze_reshaped)
        print(gaze_embedding)


        merged = Concatenate()([flatten, gaze_embedding])
        hidden = Dense(6)(merged)
        def classify(input):
            return tf.nn.softmax(input)
        def mean_value(input):
            res = tf.reshape(input,[-1,self.time_steps,num_classes])
            return tf.reduce_mean(res,1)
        # hidden = Lambda(mean_value)(hidden)
        output = Lambda(classify)(hidden)

        model = Model(input=[image, gaze], output=output)


        adam = optimizers.Adam(lr = self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['mae', 'acc'])
        print(model.summary())
        return model
    def save_model_weights(self,save_path):
		# Helper function to save your model / weights.

        self.model.save_weights(save_path)
        # self.model.save(save_path)

        # return suffix
