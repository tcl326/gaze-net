import matplotlib.pyplot as plt
import os
import sys
import numpy as np


def plot(dir_list):
    t_loss_all = []
    v_loss_all = []
    for dir in dir_list:
        print(dir)
        # get subfolder names(0,1...)
        for root, sub_dirs, files in os.walk(dir):
        	break
        for sub_dir in range(len(sub_dirs)):
            filepath = dir + str(sub_dir) + '/' + 'losses.txt'
            print(filepath)
            if not os.path.exists(filepath):
                continue
            file = open(filepath, 'r')
            t_loss = file.readline()
            v_loss = file.readline()
            file.close()

            t_loss = t_loss[1:len(t_loss)-1].split(',')
            t_loss = [float(t_loss[i][0:9]) for i in range(len(t_loss))]
            # print(t_loss)
            t_loss_all = t_loss_all + t_loss

            v_loss = v_loss[1:len(v_loss)-1].split(',')
            v_loss = [float(v_loss[i][0:9]) for i in range(len(v_loss))]
            # print(t_loss)
            v_loss_all = v_loss_all + v_loss

    # plot
    num_epoch = len(t_loss_all)
    plt.plot(range(num_epoch), t_loss_all, 'r-', range(num_epoch), v_loss_all, 'b-')
    plt.savefig('plot_loss_model2.png')

plot(['model2_bk/', 'model2/'])
