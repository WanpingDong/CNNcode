#!/usr/bin/env python
# -*- coding: utf-8 -*-
#温刚
#北京大学数学科学学院，jnwengang@pku.edu.cn
import numpy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import time
import  matplotlib.pyplot as plt
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

n_data = torch.ones(100, 2)
x_0 = torch.normal(2*n_data, 1)
y_0 = torch.zeros(100)
x_1 = torch.normal(-2*n_data, 1)
y_1 = torch.ones(100)
x = torch.cat((x_0, x_1), 0).type(torch.FloatTensor)
y = torch.cat((y_0, y_1), ).type(torch.LongTensor)

print(x.size(), y.size())
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1])
plt.show()

x, y = Variable(x), Variable(y)
net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

def differentiate(num, loss_func):
    net2 = torch.load('net.pkl')
    optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.02)
    prediction = net2(x[num: num + 1])

    loss = loss_func(prediction, y[num: num + 1])
    loss.backward()
    optimizer2.step()

    # for param in net2.parameters():
    #     print(param.grad)
    return net2.parameters()

def calInnerProduct():
    plt.cla()
    param1 = differentiate(1, loss_func)
    arr = numpy.zeros(100)
    for num in range(100):
        param2 = differentiate(num, loss_func)
        for (element1, element2) in zip(param1, param2):
            grad1_np = element1.grad.numpy()
            grad2_np = element2.grad.numpy()
            # print(grad1_np)
            print(grad2_np)
            # print("----------------------------------------------------------------------------------------")
            grad1_np = grad1_np.reshape(-1)
            grad2_np = grad2_np.reshape(-1)
            # print(grad1_np)
            # print(grad2_np)
            # print("----------------------------------------------------------------------------------------")
            print(numpy.dot(grad1_np, grad2_np))
            arr[num] = arr[num] + numpy.dot(grad1_np, grad2_np)
            # time.sleep(0.1)
        # time.sleep(0.1)
            # print(inner)
        print(arr[num])
        # print(inner)
    plt.hist(arr, bins=100, color='g')
    plt.pause(0.1)
    plt.clf()

for t in range(100):
    out = net(x)

    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 2 == 0:
        # plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        if accuracy >= 0.99:
            torch.save(net, 'net.pkl')
            calInnerProduct()

        # plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        # plt.pause(0.1)
        # plt.clf()

plt.ioff()  # 停止画图
plt.show()

