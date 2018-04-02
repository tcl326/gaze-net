
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

# global param
dataset_path = 'gaze_dataset_old/'
learning_rate = 0.0001
time_steps = 32
num_classes = 6
batch_size = 1
time_skip = 2
origin_image_size = 360    # size of the origin image before the cropWithGaze
img_size = 128    # size of the input image for network
num_channel = 3
steps_per_epoch=5
epochs=5
validation_step=20
total_num_epoch = 101


class GazeNet():
    def __init__(self,learning_rate,time_steps,num_classes,batch_size):
        self.learning_rate = 0.0001
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.model = self.create_model()

    def convolution(self, kernel_size = 5):
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
    def convolution3D(self, kernel_size = 5):
        def f(input):
            filters = 96
            conv1 = Conv3D(filters, kernel_size, strides=(3, 3, 3), padding='valid', activation=None)(input)
            conv1 = BatchNormalization()(conv1)
            conv1 = MaxPooling3D(pool_size = (2,3,3), padding = 'valid')(conv1)
            # conv1 = Dropout(0.5)(conv1)
            # conv1 = Conv3D(filters, kernel_size, strides=(2, 2, 2), padding='valid', activation=None)(conv1)
            conv1 = Dropout(0.5)(conv1)
            conv1 = Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', activation=None)(conv1)
            conv1 = MaxPooling3D(pool_size = (1,3,3),padding = 'valid')(conv1)
            return conv1
        return f

    def lstm(self):
        def f(input):
            lstm = LSTM(128,return_sequences=True,stateful = True)(input)
            lstm = Dropout(0.5)(lstm)
            lstm = LSTM(128,stateful=True)(lstm)
            return lstm
        return f
    def lstm2D(self):
        def f(input):
            lstm = LSTM(128,return_sequences=True)(input)
            lstm = Dropout(0.5)(lstm)
            lstm = LSTM(128,return_sequences=True)(lstm)
            lstm = Dropout(0.5)(lstm)
            return lstm
        return f
    def create_model(self):
        image = Input(shape=(self.time_steps,256,256,3,))
        def input_reshape(image):
            return tf.reshape(image,[-1,time_steps,256,256,3])
            # return tf.reshape(image,[self.batch_size*self.time_steps,256,256,3])
        image_reshaped = Lambda(input_reshape)(image)
        # image_embedding = self.convolution()(image_reshaped)
        image_embedding = self.convolution3D()(image_reshaped)
        image_embedding = Dropout(0.5)(image_embedding)
        flatten = Flatten()(image_embedding)

        gaze = Input(shape=(self.time_steps,3,))
        def input_gaze_reshape(input):
            return tf.reshape(input,[-1,self.time_steps,3])
        gaze_reshaped = Lambda(input_gaze_reshape)(gaze)
        # print(gaze_reshaped)
        # gaze_embedding = self.lstm()(gaze_reshaped)
        gaze_embedding = self.lstm2D()(gaze_reshaped)
        def mean_value(input):
            return tf.reduce_mean(input,1)
        gaze_embedding = Lambda(mean_value)(gaze_embedding)
        # print(gaze_embedding)


        merged = Concatenate()([flatten, gaze_embedding])
        # print(merged)
        hidden = Dense(6)(merged)
        # print(hidden)
        def classify(input):
            return tf.nn.softmax(input)
        def mean_value(input):
            res = tf.reshape(input,[self.batch_size,self.time_steps,num_classes])
            return tf.reduce_mean(res,1)
        hidden = Lambda(mean_value)(hidden)
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


	return parser.parse_args()

def train(model,pre_trained_model):
    if pre_trained_model != '':
        model.load_weights(pre_trained_model)
    for i in range(total_num_epoch):
        save_path = 'model1/'+str(i) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        trainGenerator = gaze_gen.GazeDataGenerator(validation_split=0.2)
        train_data = trainGenerator.flow_from_directory(dataset_path, subset='training',time_steps=time_steps,
                                                        batch_size=batch_size, crop=False,
                                                        gaussian_std=0.01, time_skip=time_skip, crop_with_gaze=False,
                                                       crop_with_gaze_size=128)
        val_data = trainGenerator.flow_from_directory(dataset_path, subset='validation', time_steps=time_steps,
                                                      batch_size=batch_size, crop=False,
                                                        gaussian_std=0.01, time_skip=time_skip, crop_with_gaze=False,
                                                       crop_with_gaze_size=128)
        [img_seq, gaze_seq], output = next(train_data)
        print("fetch data!")
        print(gaze_seq)
        print(img_seq.shape)
        print(gaze_seq.shape)
        print(output.shape)

        # start training
        # checkpointsString = "models/" + 'weights.{epoch:02d}  -{val_loss:.2f}.hdf5'

        # callbacks = gaze_net.save_model_weights(checkpointsString)
        # history = History()
        # checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)

        hist = model.fit_generator(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs,
                        validation_data=val_data, validation_steps=validation_step, shuffle=False)
        # input, truth = next(train_data)
        # [img_seq, gaze_seq] = input
        # predict = model.predict([img_seq, gaze_seq], batch_size=None, steps=1)

        print(hist)
        print("finished training!")
        # print(hist.history)
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
    print("test_data")
    # print(test_data)
    # loss = model.predict_generator(test_data, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False)
    # print("loss")
    # print(loss)
    # print(model.metrics_names)
    err_list = compare1(model, test_data, 49)
    print(err_list)
    # labels = np.argmax(predicted_labels,axis = 1)
    # print("labels")
    # print(labels)
def main(args):
    # generate model
    args = parse_arguments()
    gaze_net = GazeNet(learning_rate,time_steps,num_classes,batch_size)
    model = gaze_net.model
    #     plot_model(model, to_file='model.png')
    print("generate model!")
    if args.train == 1:
        train(model,args.model)
    else:
        test(model,args.model)

if __name__ == '__main__':
    main(sys.argv)
