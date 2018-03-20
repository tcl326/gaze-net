import matplotlib.pyplot as plt
import os
import sys
import numpy as np


def plot(dir_list):
    t_acc_all = []
    v_acc_all = []
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
            lines = file.readlines()[::-1]
            file.close()
            t_acc = lines[5]
            v_acc = lines[1]


            t_acc = t_acc[1:len(t_acc)-2].split(',')
            t_acc = [float(t_acc[i]) for i in range(len(t_acc))]
            # print(t_acc)
            t_acc_all = t_acc_all + t_acc

            v_acc = v_acc[1:len(v_acc)-2].split(',')
            v_acc = [float(v_acc[i]) for i in range(len(v_acc))]
            # print(t_acc)
            v_acc_all = v_acc_all + v_acc

    # print(t_acc_all)
    # plot
    num_epoch = len(t_acc_all)
    # print(num_epoch)
    # print(range(num_epoch))
    plt.plot(range(num_epoch), t_acc_all, 'r-', range(num_epoch), v_acc_all, 'b-')
    axes = plt.gca()
    # axes.set_xlim([0, 84])
    # axes.set_ylim([0, 1])
    plt.savefig('plot_acc_model1.png')

plot(['model1/'])
