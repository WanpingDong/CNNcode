import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
show = ToPILImage() #可以把Tensor转成Image,方便可视化


# torchvision是独立于pytorch的关于图像操作的一些方便工具库。
# vision.datasets : 几个常用视觉数据集，可以下载和加载
# vision.models : 流行的模型，例如 AlexNet, VGG, ResNet 和 Densenet 以及训练好的参数。
# vision.transforms : 常用的图像操作，例如：数据类型转换，图像到tensor ,numpy 数组到tensor , tensor 到 图像等。
# vision.utils : 用于把形似 (3 x H x W) 的张量保存到硬盘中，给一个mini-batch的图像可以产生一个图像格网

transform=transforms.Compose([
    transforms.ToTensor(),#转为Tensor
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#归一化
])


# 定义一些超参数
BATCH_SIZE = 32  # batch_size即每批训练的样本数量
EPOCHS = 100  # 循环次数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，即device定义为CUDA或CPU

# 下载 CIFAR 的数据集

CIFAR_DATA = datasets.CIFAR10('data',
                              train=True,
                              download=False,
                              transform=transform
                              )

# 训练集
train_loader = torch.utils.data.DataLoader(  # vision.utils : 用于把形似 (3 x H x W) 的张量保存到硬盘中，给一个mini-batch的图像可以产生一个图像格网。
    CIFAR_DATA,
    batch_size=BATCH_SIZE,
    shuffle=True)  # shuffle() 方法将序列的所有元素随机排序

# 测试集
test_loader = torch.utils.data.DataLoader(CIFAR_DATA, batch_size=BATCH_SIZE, shuffle=True)

classes=('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

data_label0, data_label1, data_label2 = eval(input('Please Enter 3 Numbers between 0 and 9 Separated by Commas: '))

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

# -------------标签为data_label0的train_data----------------------------------
mask0 = [1 if CIFAR_DATA[i][1] == data_label0 else 0 for i in range(len(CIFAR_DATA))]
mask0 = torch.tensor(mask0)
sampler0 = MyDataSet0(mask0, CIFAR_DATA)
train_loader0 = torch.utils.data.DataLoader(CIFAR_DATA,
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

# -------------标签为data_label2的train_data----------------------------------
mask1 = [1 if CIFAR_DATA[i][1] == data_label1 else 0 for i in range(len(CIFAR_DATA))]
mask1 = torch.tensor(mask1)
sampler1 = MyDataSet1(mask1, CIFAR_DATA)
train_loader1 = torch.utils.data.DataLoader(CIFAR_DATA,
                                            batch_size=BATCH_SIZE,
                                            sampler=sampler1,
                                            shuffle=False)

class MyDataSet2(torch.utils.data.sampler.Sampler):
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(mask2)])

    def __len__(self):
        return len(self.data_source)

# -------------标签为data_label2的train_data----------------------------------
mask2 = [1 if CIFAR_DATA[i][1] == data_label2 else 0 for i in range(len(CIFAR_DATA))]
mask2 = torch.tensor(mask2)
sampler2 = MyDataSet2(mask2, CIFAR_DATA)
train_loader2 = torch.utils.data.DataLoader(CIFAR_DATA,
                                            batch_size=BATCH_SIZE,
                                            sampler=sampler2,
                                            shuffle=False)

# 下面我们定义一个网络，网络包含两个卷积层，conv1和conv2，
# 然后紧接着三个线性层作为输出，
# 最后输出10个维度，这10个维度我们作为0-9的标识来确定识别出的是哪个类

# 这里建议大家将每一层的输入和输出维度都作为注释标注出来，这样后面阅读代码的会方便很多

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 我们实例化一个网络，实例化后使用“.to”方法将网络移动到GPU
model = ConvNet().to(DEVICE)

# 优化器我们也直接选择简单暴力的 SGD
loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def differentiate(data, target):
    net2 = torch.load('model.pkl')
    optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.001, momentum=0.9)
    # loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数
    output = net2(data)

    gradient_shape = torch.zeros_like(output)
    gradient_shape[:, data_label0] = 1

    # loss = loss_func(output, target)
    # loss.backward()

    output.backward(gradient_shape)
    optimizer2.step()

    # for param in net2.parameters():
    #     print(param.grad)
    return net2.parameters()

def calInnerProduct2(device):
    plt.cla()
    param1 = differentiate(data0, target0)
    param1 = list(param1)
    arr = numpy.zeros(len(list(train_loader0)))
    for batch_idx, (data, target) in enumerate(train_loader0):
        data, target = data.to(device), target.to(device)  # CPU转GPU
        param2 = differentiate(data, target)
        # param2 = list(param2)
        for element1, element2 in zip(param1, param2):
            grad1_np = element1.grad.numpy()
            grad2_np = element2.grad.numpy()
            # print("----------------------------------------------------------------------------------------")
            grad1_np = grad1_np.reshape(-1)
            grad2_np = grad2_np.reshape(-1)
            # print("----------------------------------------------------------------------------------------")
            arr[batch_idx] = arr[batch_idx] + numpy.dot(grad1_np, grad2_np)
    plt.hist(arr, bins=10, color='g')

    plt.pause(0.1)
    plt.clf()

def calInnerProduct(device):
    # plt.cla()
    # fig = plt.figure(figsize=(16, 12))
    # ax1 = fig.add_subplot(111)
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
        arr0[batch_idx] = arr0[batch_idx] / (math.sqrt(norm1) * math.sqrt(norm2))

    # ax1.hist(arr0, bins=10, color='gold', alpha=0.7,
    #          label='the inner product of {} and {}'.format(data_label0, data_label1))
    plt.hist(arr0, bins=10, color='gold', alpha=0.7,
             label='the inner product of {} and {}'.format(data_label0, data_label1))
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
        arr[batch_idx] = arr[batch_idx] / (math.sqrt(norm1) * math.sqrt(norm2))

    plt.hist(arr, bins=10, color='salmon', alpha=0.7,
             label='the inner product of {} and {}'.format(data_label0, data_label2))
    plt.legend()
    # plt.show()
    plt.pause(0.1)
    plt.clf()

    # ax1.hist(arr, bins=10, color='salmon', alpha=0.7,
    #          label='the inner product of {} and {}'.format(data_label0, data_label2))
    # ax1.legend()

    # plt.savefig('temp.png')
    # plt.close()
    # image_list.append(imageio.imread('temp.png'))
    # plt.clf()


# 定义 训练函数 ，我们将训练的所有操作都封装到train函数中
def train(model, device, train_loader, optimizer, epoch):
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # CPU转GPU
        optimizer.zero_grad()  # 优化器清零
        output = model(data)  # 由model，计算输出值
        loss = loss_func(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx + 1) % 2000 == 0:
            print('epoch:', epoch + 1, '|i:', batch_idx + 1, '|loss:%.3f' % (running_loss / 2000))
            running_loss = 0.0



# ---------------------测试函数------------------------------
# 测试的操作也一样封装成一个函数
def test(model, device, test_loader):
    correct = 0  # 定义的预测正确的图片数
    total = 0  # 总共图片个数
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # CPU -> GPU
            output = model(data)
            _, predict = torch.max(output, 1)
            total += target.size(0)
            correct += (predict == target).sum()

    print('测试集中的准确率为：%d%%' % (100 * correct / total))
    if correct / total > 0.1:
        torch.save(model, 'model.pkl')
        calInnerProduct(device)


# ---------------------------------------------------------------

# 下面开始训练，这里就体现出封装起来的好处了，只要写两行就可以了
# 整个数据集只过一遍

for epoch in range(EPOCHS):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)

