#!/usr/bin/env python
# -*- coding: utf-8 -*-
#温刚
#北京大学数学科学学院，jnwengang@pku.edu.cn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy
import math
# import cupy as numpy
from torchvision import datasets, transforms
import imageio
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torchvision是独立于pytorch的关于图像操作的一些方便工具库。
# vision.datasets : 几个常用视觉数据集，可以下载和加载
# vision.models : 流行的模型，例如 AlexNet, VGG, ResNet 和 Densenet 以及训练好的参数。
# vision.transforms : 常用的图像操作，例如：数据类型转换，图像到tensor ,numpy 数组到tensor , tensor 到 图像等。
# vision.utils : 用于把形似 (3 x H x W) 的张量保存到硬盘中，给一个mini-batch的图像可以产生一个图像格网

# print(torch.__version__)  # 检查 pytorch 的版本

# 定义一些超参数
BATCH_SIZE = 32  # batch_size即每批训练的样本数量
EPOCHS = 10  # 循环次数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，即device定义为CUDA或CPU

# 下载 MNIST的数据集
# 训练集

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

MNIST_DATA = datasets.MNIST('data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),  # 图像转化为Tensor
                       transforms.Normalize((0.1307,), (0.3081,))  # 标准化（参数不明）
                   ]))


train_loader = torch.utils.data.DataLoader(  # vision.utils : 用于把形似 (3 x H x W) 的张量保存到硬盘中，给一个mini-batch的图像可以产生一个图像格网。
    datasets.MNIST('data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),  # 图像转化为Tensor
                       transforms.Normalize((0.1307,), (0.3081,))  # 标准化（参数不明）
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)  # shuffle() 方法将序列的所有元素随机排序


# 测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=BATCH_SIZE, shuffle=True)  # shuffle() 方法将序列的所有元素随机排序

data_label0, data_label1, data_label2 = eval(input('Please Enter 3 Numbers Seperated by commas: '))

# 筛选标签为0/1的数据

# 定义自己的 Dataset
class MyDataSet0(torch.utils.data.sampler.Sampler):
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(mask0)])

    def __len__(self):
        return len(self.data_source)

# -------------标签为0的train_data----------------------------------
mask0 = [1 if MNIST_DATA[i][1] == data_label0 else 0 for i in range(len(MNIST_DATA))]
mask0 = torch.tensor(mask0).cuda()
sampler0 = MyDataSet0(mask0, MNIST_DATA)
train_loader0 = torch.utils.data.DataLoader(MNIST_DATA,
                                            batch_size=BATCH_SIZE,
                                            sampler=sampler0,
                                            shuffle=False)

train_data0 = enumerate(train_loader0).__next__()[1]
data0 = train_data0[0]
target0 = train_data0[1]
data0, target0 = data0.to(DEVICE), target0.to(DEVICE)

# 定义自己的 Dataset
class MyDataSet1(torch.utils.data.sampler.Sampler):
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(mask1)])

    def __len__(self):
        return len(self.data_source)

# -------------标签为1的train_data----------------------------------
mask1 = [1 if MNIST_DATA[i][1] == data_label1 else 0 for i in range(len(MNIST_DATA))]
mask1 = torch.tensor(mask1).cuda()
sampler1 = MyDataSet1(mask1, MNIST_DATA)

train_loader1 = torch.utils.data.DataLoader(MNIST_DATA,
                                            batch_size=BATCH_SIZE,
                                            sampler=sampler1,
                                            shuffle=False)

# 定义自己的 Dataset
class MyDataSet2(torch.utils.data.sampler.Sampler):
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(mask2)])

    def __len__(self):
        return len(self.data_source)

# -------------标签为1的train_data----------------------------------
mask2 = [1 if MNIST_DATA[i][1] == data_label2 else 0 for i in range(len(MNIST_DATA))]
mask2 = torch.tensor(mask2).cuda()
sampler2 = MyDataSet2(mask2, MNIST_DATA)
train_loader2 = torch.utils.data.DataLoader(MNIST_DATA,
                                            batch_size=BATCH_SIZE,
                                            sampler=sampler2,
                                            shuffle=False)


# 下面我们定义一个网络，网络包含两个卷积层，conv1和conv2，
# 然后紧接着两个线性层作为输出，
# 最后输出10个维度，这10个维度我们作为0-9的标识来确定识别出的是那个数字

# 这里建议大家将每一层的输入和输出维度都作为注释标注出来，这样后面阅读代码的会方便很多


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 128x28
        self.conv1 = nn.Conv2d(1, 10, 5)  # 10, 24x24
        self.conv2 = nn.Conv2d(10, 20, 3)  # 128, 10x10
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)  # in_size 为 batch_size（一个batch中的Sample数）
        # 卷积层 -> relu -> 最大池化
        out = self.conv1(x)  # 24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 12
        # 卷积层 -> relu -> 多行变一行 -> 全连接层 -> relu -> 全连接层 -> sigmoid
        out = self.conv2(out)  # 10
        out = F.relu(out)
        out = out.view(in_size, -1)  # view()函数作用是将一个多行的Tensor,拼接成一行。
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        # softmax
        out = F.log_softmax(out, dim=1)
        # 返回值 out
        return out


# 我们实例化一个网络，实例化后使用“.to”方法将网络移动到GPU
model = ConvNet().to(DEVICE)

# 优化器我们也直接选择简单暴力的 SGD
optimizer = optim.SGD(model.parameters(), lr=0.02)


def differentiate(data, target):
    net2 = torch.load('model.pkl')
    optimizer2 = optim.SGD(net2.parameters(), lr=0.02)
    optimizer2.zero_grad()  # 优化器清零
    output = net2(data)
    loss = F.nll_loss(output, target)  # 计算损失函数loss
    loss.backward()  # loss反向传播
    optimizer2.step()

    # for param in net2.parameters():
    #     print(param.grad)
    return net2.parameters()

def calInnerProduct(device):
    # plt.cla()
    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(111)
    param1 = differentiate(data0, target0)
    param1 = list(param1)
    arr0 = numpy.zeros(len(list(train_loader1)))
    for batch_idx, (data, target) in enumerate(train_loader1):
        # print(batch_idx)
        data, target = data.to(device), target.to(device)  # CPU转GPU
        param2 = differentiate(data, target)
        param2 = list(param2)
        # param1, param2 = param1.cpu(), param2.cpu()
        norm1 = 0
        norm2 = 0
        for element1, element2 in zip(param1, param2):
            grad1_np = element1.grad.cpu().numpy()
            grad2_np = element2.grad.cpu().numpy()
            # print("----------------------------------------------------------------------------------------")
            grad1_np = grad1_np.reshape(-1)
            grad2_np = grad2_np.reshape(-1)
            # print("----------------------------------------------------------------------------------------")
            arr0[batch_idx] = arr0[batch_idx] + numpy.dot(grad1_np, grad2_np)
            norm1 = norm1 + numpy.dot(grad1_np, grad1_np)
            norm2 = norm2 + numpy.dot(grad2_np, grad2_np)
        arr0[batch_idx] = arr0[batch_idx]/(math.sqrt(norm1) * math.sqrt(norm2))

    ax1.hist(arr0, bins=10, color='gold', alpha=0.7, label='the inner product of {} and {}'.format(data_label0, data_label1))


    param1 = differentiate(data0, target0)
    param1 = list(param1)
    arr = numpy.zeros(len(list(train_loader2)))
    for batch_idx, (data, target) in enumerate(train_loader2):
        # print(batch_idx)
        data, target = data.to(device), target.to(device)  # CPU转GPU
        param2 = differentiate(data, target)
        param2 = list(param2)
        # param1, param2 = param1.cpu(), param2.cpu()
        norm1 = 0
        norm2 = 0
        for element1, element2 in zip(param1, param2):
            grad1_np = element1.grad.cpu().numpy()
            grad2_np = element2.grad.cpu().numpy()
            # print("----------------------------------------------------------------------------------------")
            grad1_np = grad1_np.reshape(-1)
            grad2_np = grad2_np.reshape(-1)
            # print("----------------------------------------------------------------------------------------")
            arr[batch_idx] = arr[batch_idx] + numpy.dot(grad1_np, grad2_np)
            norm1 = norm1 + numpy.dot(grad1_np, grad1_np)
            norm2 = norm2 + numpy.dot(grad2_np, grad2_np)
        arr[batch_idx] = arr[batch_idx]/(math.sqrt(norm1) * math.sqrt(norm2))

    ax1.hist(arr, bins=10, color='salmon', alpha=0.7, label='the inner product of {} and {}'.format(data_label0, data_label2))
    ax1.legend()

    plt.savefig('temp.png')
    plt.close()
    image_list.append(imageio.imread('temp.png'))
    # plt.clf()


# 定义 训练函数 ，我们将训练的所有操作都封装到train函数中
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # CPU转GPU
        optimizer.zero_grad()  # 优化器清零
        output = model(data)  # 由model，计算输出值
        # print(data.size())
        loss = F.nll_loss(output, target)  # 计算损失函数loss
        loss.backward()  # loss反向传播
        optimizer.step()  # 优化器优化
        if (batch_idx + 1) % 1000 == 0:  # 输出结果
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))




# -------------------------------------------------------------

# ---------------------测试函数------------------------------
# 测试的操作也一样封装成一个函数
def test(model, device, test_loader):
    test_loss = 0  # 损失函数初始化为0
    correct = 0  # correct 计数分类正确的数目
    with torch.no_grad():  # 表示不反向求导（反向求导为训练过程）
        for data, target in test_loader:  # 遍历所有的data和target
            data, target = data.to(device), target.to(device)  # CPU -> GPU
            output = model(data)  # output为预测值，由model计算出
            test_loss += F.nll_loss(output, target, reduction='sum').item()  ### 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  ### 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)  # 总损失除数据集总数
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if correct / len(test_loader.dataset) > 0.95:
        torch.save(model, 'model.pkl')
        calInnerProduct(device)


# ---------------------------------------------------------------

# 下面开始训练，这里就体现出封装起来的好处了，只要写两行就可以了
# 整个数据集只过一遍

image_list = []
for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
name = '178'
imageio.mimsave(name+'.gif', image_list, duration=0.4)
