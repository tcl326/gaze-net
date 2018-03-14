import os
import sys
import tensorflow as tf
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense,LSTM
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, TimeDistributed, Lambda
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, RepeatVector
from keras.layers import Activation, BatchNormalization, MaxPooling2D
import time
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import Input
from keras import backend as K
K.set_image_dim_ordering('tf')
import math
import gazenetGenerator as gaze

learning_rate = 0.0001
timesteps = 32
num_classes = 6
batch_size = 4
# size of the origin image before the cropWithGaze
origin_image_size = 360
# size of the input image for network
img_size = 128
num_channel = 3

class GazeNet():
    def __init__(self,learning_rate,timesteps,num_classes,batch_size)
        self.learning_rate = learning_rate
        self.timesteps = timesteps
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.gaussian_sigma = 1
        self.gaussian_weight = self.create_gaussian_weight()
        self.model = self.create_mode()

    def create_gaussian_weight(self):
        kernel_size = 5    #same with the shape of the layer before flatten
        kernel_num = 256
        r = (kernel_size - 1) // 2
        sigma_2 = float(self.gaussian_sigma * self.gaussian_sigma)
        pi = 3.1415926
        ratio = 1 / (2*pi*sigma_2)

        kernel = np.zeros((kernel_size, kernel_size))
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                tmp = math.exp(-(i*i+j*j)/(2*sigma_2))
                kernel[i+r][j+r] = round(tmp, 3)
        kernel *= ratio
        kernel = np.expand_dims(kernel, axis=2)
        kernel = np.tile(kernel, (1,1,kernel_num))
        # print(kernel.shape)
        return kernel

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

        def multiply_constant(input):
            for i in range(self.batch_size*self.timesteps):
                tmp = tf.multiply(tf.cast(input[i], tf.float32), tf.cast(self.gaussian_weight, tf.float32))
                tmp = tf.expand_dims(tmp, 0)
                if i == 0:
                    res = tmp
                else:
                    res = tf.concat([res, tmp], 0)
            res = tf.reshape(res,[self.batch_size,self.timesteps,5,5,256])
            res = tf.reshape(res,[self.batch_size,self.timesteps,6400])
            return res

        model.add(Lambda(multiply_constant))

        def mean_value(input):
            return tf.reduce_mean(input,1)

        model.add(LSTM(128,return_sequences = True))
        model.add(LSTM(6,return_sequences = True))
        model.add(Lambda(mean_value))

        adam = optimizers.Adam(lr = self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # lstmModel.compile(loss = 'categorical_crossentropy',optimizer = 'adam')
        print(model.summary())

        return model


def cropWithGaze(images,gazes,batch_size,timesteps,img_size,num_channel):
    img_seq = np.zeros(batch_size,timesteps,img_size,img_size,num_channel)
    for i in range(batch_size):
        for j in range(timesteps):
            x = gaze[1]
            y = gaze[2]
            image_size = images.shape[2]
            assert image_size > img_size
            right_bound = min(x+(img_size/2),image_size)
            left_bound = max(0,x-(img_size/2))
            up_bound = max(0,y-(img_size/2))
            down_bound = min(image_size,y+(img_size/2))
            img_seq(batch_size,timesteps,:,:,:) = images(batch_size,timesteps,up_bound:down_bound,left_bound:right_bound,:)
    return img_seq


def main(args):
    gaze_net = GazeNet(learning_rate,timesteps,num_classes,batch_size)
    trainGenerator = gaze.GazeDataGenerator()
    trainGeneratorDirectory = trainGenerator.flow_from_directory('../gaze_dataset', time_steps=10, batch_size=2)
    [img_seq, gaze_seq], output = next(trainGeneratorDirectory)
    img_seq = cropWithGaze(img_seq,gaze_seq,batch_size,timesteps,img_size,num_channel)

    image_input = Input(shape=(224, 224, 3))


if __name__ == '__main__':
	main(sys.argv)
