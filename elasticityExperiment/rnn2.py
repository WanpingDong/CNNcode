#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 温刚
# 北京大学数学科学学院，jnwengang@pku.edu.cn

import torch
import utils2
from torch.autograd import Variable
import random

T = 100

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0.01)


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 数据点横坐标
# y = x+0.1*torch.randn(x.size())
y = x.pow(4) + 0.1 * torch.randn(x.size())  # 数据点对应的纵坐标
x, y = (Variable(x), Variable(y))  # training set, pytorch handlable

width = 100
depth = 2
kind = 'fully connected'

net = torch.nn.Sequential(
    torch.nn.Linear(1, width),
    torch.nn.Sigmoid(),
    torch.nn.Linear(width, width),
    torch.nn.Sigmoid(),
    # torch.nn.Linear(width, width),
    # torch.nn.ReLU(),
    # torch.nn.Linear(width, width),
    # torch.nn.ReLU(),
    # torch.nn.Linear(width, width),
    # torch.nn.ReLU(),
    # torch.nn.Linear(width, width),
    # torch.nn.ReLU(),
    # torch.nn.Linear(width, width),
    # torch.nn.ReLU(),
    torch.nn.Linear(width, 1)
)  # 全连接神经网络
# net[2 * depth - 4].register_forward_hook(live_relu_hook)
net.apply(init_weights)
print(net)
print(net.parameters())
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

outputs = []
def hook(module, input, output):
    outputs.append(output.clone().detach())

for t in range(T):  # 迭代5000次
    i = random.randint(0, 99)

    unrelated = 99 - i

    outputs = []

    handle = net[len(net)-2].register_forward_hook(hook)  ## 获取所有x对整个net模型倒数第二层的中间结果
    prediction0 = net(x)  # 先用网络对所有的x预测一次,是一个数组

    # print(outputs[0].size())
    # print(len(outputs))
    handle.remove()

    prediction = net(x[i:i + 1])  # SGD选择的样本

    if t == 0:
        prediction0_table = torch.unsqueeze(prediction0, 0)
        selected_table = torch.unsqueeze(x[i:i + 1], 0)
        selectedy_table = torch.unsqueeze(y[i:i + 1], 0)  # prediction 表
        outputs_sgd = torch.unsqueeze(outputs[0][i: i + 1, :], 0)
        outputs_unrelated = torch.unsqueeze(outputs[0][unrelated: unrelated + 1, :], 0)
    else:
        prediction0_table = torch.cat((prediction0_table, torch.unsqueeze(prediction0, 0)), dim=0)  # prediction0 表
        selected_table = torch.cat((selected_table, torch.unsqueeze(x[i:i + 1], 0)), dim=0)  # prediction 表
        selectedy_table = torch.cat((selectedy_table, torch.unsqueeze(y[i:i + 1], 0)), dim=0)  # prediction 表
        outputs_sgd = torch.cat((outputs_sgd, torch.unsqueeze(outputs[0][i:i + 1, :], 0)), dim=0)  # outputs_sgd 表
        outputs_unrelated = torch.cat((outputs_unrelated, torch.unsqueeze(outputs[0][unrelated: unrelated + 1, :], 0)),
                                      dim=0)
    loss = loss_func(prediction, y[i:i + 1])  # 我们计算SGD选出的样本的损失

    optimizer.zero_grad()  # 把net全部的梯度归零
    loss.backward()  # 向后传播
    optimizer.step()

    prediction = net(x)  # 更新后的再预测一次
    if t == 0:
        prediction_table = torch.unsqueeze(prediction, 0)
    else:
        prediction_table = torch.cat((prediction_table, torch.unsqueeze(prediction, 0)), dim=0)  # prediction 表

    q = abs(prediction - prediction0)

    q1 = sorted(q)  # 把q排序
    threshold = q1[len(q1) - 5]  # 选出五个最大的预测值的变化
    # print(threshold)
    loss = loss_func(prediction, y)  # 计算总损失

    q = (torch.ge(q, threshold)).nonzero()
    q = q[0:5, 0]

    # print(q)

    if t == 0:
        q_table = torch.unsqueeze(q, 0)
        outputs_q = torch.unsqueeze(outputs[0][q[0]: q[0] + 1, :], 0)
        for i in range(len(q)-1):
            outputs_q = torch.cat((outputs_q, torch.unsqueeze(outputs[0][q[i+1]: q[i+1]+1, :], 0)), dim=0)
    else:
        # print(q_table.shape, q)
        q_table = torch.cat((q_table, torch.unsqueeze(q, 0)), dim=0)  # q 表
        for i in q:
            outputs_q = torch.cat((outputs_q, torch.unsqueeze(outputs[0][i: i+1, :], 0)), dim=0)


utils2.plotdif(T, x, y, prediction_table, prediction0_table, selected_table, selectedy_table, q_table, outputs_sgd, outputs_q, outputs_unrelated, name = f'Sigmoid_wide_deep_{depth}_width_{width}_point')
