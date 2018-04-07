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
import pdb
import gazeWholeGenerator as gaze_gen_whole
import gazenetGenerator as gaze_gen
from compare_predict_truth import compare1
import theano
# global param
dataset_path = 'gaze_dataset_old/'
learning_rate = 0.00001
time_steps = 32
num_classes = 6
batch_size = 4
time_skip = 2
origin_image_size = 360    # size of the origin image before the cropWithGaze
img_size = 128    # size of the input image for network
num_channel = 3
steps_per_epoch=161
epochs=1
validation_step=20
total_num_epoch = 101
from random import randint



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
    parser.add_argument('--model_type',dest='model_type',type=str,default='GazeNet_native',
                        help='choose from GazeNet_2D, GazeNet_3D, GazeNet_native, GazeNet_sequential_2D')
    return parser.parse_args()

def train(model,args):

    pre_trained_model = args.model
    if pre_trained_model != '':
        model.load_weights(pre_trained_model)
    if args.model_type == 'GazeNet_sequential_2D':
        batch_size=1
        time_steps = 400
        trainGenerator = gaze_gen_whole.GazeDataGenerator(validation_split=0.2)
        dataset_path = 'gaze_dataset/'

    elif args.model_type == 'GazeNet_3D' or args.model_type == 'GazeNet_2D' or args.model_type == 'GazeNet_native':
        batch_size=4
        time_steps=32
        trainGenerator = gaze_gen.GazeDataGenerator(validation_split=0.2)
        dataset_path = 'gaze_dataset_old'
    else:
        raise ValueError('invalid model type name' )

    train_data = trainGenerator.flow_from_directory(dataset_path, subset='training',time_steps=time_steps,
                                                    batch_size=batch_size, crop=False,
                                                    gaussian_std=0.01, time_skip=time_skip)
    val_data = trainGenerator.flow_from_directory(dataset_path, subset='validation', time_steps=time_steps,
                                                  batch_size=batch_size, crop=False,
                                                    gaussian_std=0.01, time_skip=time_skip)
    for i in range(total_num_epoch):
        print(i)
        save_path = 'model1/'+ args.model_type + '/'+str(i) + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.000001)
        pdb.set_trace()
        hist = model.fit_generator(train_data, steps_per_epoch=steps_per_epoch, epochs=epochs, max_queue_size=1,
                        validation_data=val_data, validation_steps=validation_step, callbacks=[reduce_lr],shuffle=False)
        confusion_matrix_train = np.zeros((num_classes,num_classes))
        pdb.set_trace()
        print(convout1_f)
        for i in range(28):
            [img_seq,gaze_seq],label = next(train_data)
            y = model.predict([img_seq,gaze_seq],batch_size = 1)
            predicted_label = y.argmax(axis=1)
            grounded_label = label.argmax(axis=1)
            for j in range(len(predicted_label)):
                confusion_matrix_train[predicted_label[j],grounded_label[j]] = confusion_matrix_train[predicted_label[j],grounded_label[j]]+1
                if predicted_label[j]!=grounded_label[j]:
                    print(predicted_label,grounded_label)

        confusion_matrix_val = np.zeros((num_classes,num_classes))
        for i in range(7):
            [img_seq,gaze_seq],label = next(val_data)
            y = model.predict([img_seq,gaze_seq],batch_size = 1)
            predicted_label = y.argmax(axis=1)
            grounded_label = label.argmax(axis=1)
            for j in range(len(predicted_label)):
                confusion_matrix_val[predicted_label[j],grounded_label[j]] = confusion_matrix_val[predicted_label[j],grounded_label[j]]+1
                if predicted_label[j]!=grounded_label[j]:
                    print(predicted_label,grounded_label)


        print("confusion_matrix_train")
        print(confusion_matrix_train)
        print("confusion_matrix_val")
        print(confusion_matrix_val)

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
    gaze_net = GazeNet_native(learning_rate,time_steps,num_classes,batch_size)
    model = gaze_net.model
    print("generate model!")
    if args.train == 1:
        train(model,args)
    else:
        test(model,args.model)

if __name__ == '__main__':
    main(sys.argv)
