#!/usr/bin/env python
# -*- coding: utf-8 -*-
#温刚
#北京大学数学科学学院，jnwengang@pku.edu.cn
import torch
import os
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from torch.autograd import Variable
import imageio
import numpy

plt.switch_backend('agg')
pointsize = 1
def plotdif(T, x, y, prediction_table, prediction0_table, selected_table, selectedy_table, q_table, outputs_sgd, outputs_q, outputs_unrelated, name):
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    prediction_table = prediction_table.data.cpu().numpy()
    prediction0_table = prediction0_table.data.cpu().numpy()
    selected_table = selected_table.data.cpu().numpy()
    selectedy_table = selectedy_table.data.cpu().numpy()
    outputs_q = outputs_q.data.cpu().numpy()
    outputs_sgd = outputs_sgd.data.cpu().numpy()
    outputs_unrelated = outputs_unrelated.data.cpu().numpy()
    # q_table = q_table.data.numpy()



    image_list = []
    for t in range(T):
        fig = plt.figure(figsize=(16,12))
        ax1 = fig.add_subplot(121)

        ax1.scatter(x, y, color='yellow')  # 用黄色把基准函数画出来
        # ax.plot(label = f'{width}')
        ax1.scatter(selected_table[t], selectedy_table[t])  # 把SGD选出来的点标出来
        ax1.plot(x, prediction_table[t], 'r-', lw=5)
        ax1.plot(x, prediction0_table[t], 'g-', lw=5)

        ax2 = fig.add_subplot(122)
        index_sgd = (numpy.maximum(outputs_sgd[t], 0)).nonzero()
        index_sgd = index_sgd[1]
        # sgd_row = numpy.ones_like(index_sgd)
        sgd_row = numpy.ones_like(index_sgd)
        ax2.scatter(sgd_row, index_sgd, s=pointsize)
        ax2.set_xticks([])
        #
        lines = []
        for i in range(t*5, t*5+5):
            index_q = (numpy.maximum(outputs_q[i], 0)).nonzero()
            index_q = index_q[1]
            inter = numpy.intersect1d(index_sgd, index_q)
            rate = len(inter) / len(index_sgd)

            q_row = numpy.ones_like(index_q) * (i%5+2)

            l = ax2.scatter(q_row, index_q, s=pointsize, color='black', label=f'rate={rate}')
            lines.append(l)
            # labels = [l.get_label() for l in lines]
            # ax2.legend(lines, labels, loc=(1, 1))


        index_unr = (numpy.maximum(outputs_unrelated[t], 0)).nonzero()
        index_unr = index_unr[1]
        inter = numpy.intersect1d(index_sgd, index_unr)
        rate = len(inter) / len(index_sgd)

        unr_row = numpy.ones_like(index_unr) * 7
        l = ax2.scatter(unr_row, index_unr, s=pointsize, color='orange', label=f'rate={rate}')
        lines.append(l)
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc=(1, 1))

        id = q_table[t]
        for i in id:
            ax1.scatter(x[i:i+1], y[i:i+1], color='black')
        # plt.legend()


        plt.savefig('temp.png')
        plt.close()
        image_list.append(imageio.imread('temp.png'))


    imageio.mimsave(name+'.gif', image_list, duration=0.4)
    # plt.ioff()
    # plt.show()

def plot_sgd_gd(T,x_test, y_test, x, y, prediction_table, prediction0_table, selected_table, selectedy_table, q_table, name):
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    x_test = x_test.data.cpu().numpy()
    y_test = y_test.data.cpu().numpy()
    prediction_table = prediction_table.data.cpu().numpy()
    prediction0_table = prediction0_table.data.cpu().numpy()
    selected_table = selected_table.data.cpu().numpy()
    selectedy_table = selectedy_table.data.cpu().numpy()



    image_list = []
    for t in range(T):
        fig = plt.figure(figsize=(16,12))
        ax1 = fig.add_subplot(111)

        ax1.scatter(x_test, y_test, color='yellow')  # 用黄色把基准函数画出来
        # ax.plot(label = f'{width}')
        ax1.scatter(selected_table[t], selectedy_table[t])  # 把SGD选出来的点标出来
        ax1.plot(x_test, prediction_table[t], 'r-', lw=5)
        ax1.plot(x_test, prediction0_table[t], 'g-', lw=5)

        id = q_table[t]
        for i in id:
            # print(i)
            ax1.scatter(x[i:i+1], y[i:i+1], color='black')
        # plt.legend()


        plt.savefig('temp.png')
        plt.close()
        image_list.append(imageio.imread('temp.png'))


    imageio.mimsave(name+'.gif', image_list, duration=0.4)
    # plt.ioff()
    # plt.show()
