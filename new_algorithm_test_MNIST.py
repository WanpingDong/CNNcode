import numpy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import os
import random
import torch.utils.data.sampler as sampler
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
ids = [0, 1]
torch.manual_seed(0)

# 定义一些超参数
BATCH_SIZE = 128  # batch_size即每批训练的样本数量
EPOCHS = 100  # 循环次数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，即device定义为CUDA或CPU
CLASSES = 10
Learning_Rate_New = 0.01
Learning_Rate_SGD = 0.01


train_set = torchvision.datasets.MNIST(root='data', train=True, download=False,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),  # 图像转化为Tensor
                                           transforms.Normalize((0.1307,), (0.3081,))  # 标准化（参数不明）
                                       ]))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)


# -------------标签为data_label的train_data----------------------------------
class MySampler(torch.utils.data.sampler.Sampler):
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        mask0 = [1 if train_set[i][1] == self.mask else 0 for i in range(len(train_set))]
        mask0 = torch.tensor(mask0)
        return iter([i.item() for i in torch.nonzero(mask0)])

    def __len__(self):
        return int(len(self.data_source) / CLASSES)

train_loader_list = []
for data_label in range(CLASSES):
    sampler0 = MySampler(data_label, train_set)
    train_loader_part = torch.utils.data.DataLoader(train_set,
                                                    batch_size=BATCH_SIZE,
                                                    sampler=sampler0,
                                                    shuffle=False)
    train_loader_list.append(train_loader_part)

data_set = []
for data_label in range(CLASSES):
    data_set.append(next(iter(train_loader_list[data_label])))

index = [i for i in range(CLASSES)]

class Net(nn.Module):
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

    def partial_grad(self, data, target, criterion):
        """
        Function to compute the grad
        args : data, target, loss_function
        return loss
        """
        outputs = self.forward(data)
        loss = criterion(outputs, target)
        loss.backward()
        return loss

    def calculate_loss_grad(self, dataset, criterion, n_samples):
        """
        Function to compute the full loss and the full gradient
        args : dataset, loss function and number of samples
        return : total loss and full grad norm
        """
        total_loss = 0.0
        full_grad = []
        for inputs, labels in dataset:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            total_loss += (1./n_samples) * self.partial_grad(inputs, labels, criterion)

        for param in self.parameters():
            full_grad.append((1. / n_samples) * param.grad)

        return total_loss, full_grad

    def cal_inner_product(self, grad, param):
        inner_product = 0.0
        norm_grad = 0.0
        for element1, element2 in zip(grad, param):
            grad1_np = copy.deepcopy(element1)
            grad2_np = copy.deepcopy(element2.grad)
            grad1_np = grad1_np.view(-1)
            grad2_np = grad2_np.view(-1)
            inner_product += torch.dot(grad1_np, grad2_np)
            norm_grad += torch.dot(grad1_np, grad1_np)

        return inner_product / norm_grad

    def newalg_backward(self, criterion, n_epoch, lr):
        """
        Function to updated weights with a SVRG backpropagation
        args : dataset, loss function, number of epochs, learning rate
        return : total_loss_epoch, grad_norm_epoch
        """
        total_loss_epoch = [0 for i in range(n_epoch)]
        for epoch in range(n_epoch):
            print(epoch)

            random.shuffle(index)
            for data_label in index:
                # Compute full grad
                self.zero_grad()
                total_loss_epoch[epoch], full_grad = \
                    self.calculate_loss_grad(data_set, criterion, CLASSES)

                inputs, labels = data_set[data_label]
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # Compute cur stoc grad
                self.zero_grad()
                loss = self.partial_grad(inputs, labels, criterion)

                # Backward
                for grad, param in zip(full_grad, self.parameters()):
                    grad.data -= param.grad.data / CLASSES
                inner_product = self.cal_inner_product(full_grad, self.parameters())
                for grad, param in zip(full_grad, self.parameters()):
                    param.grad.data -= inner_product * grad.data
                    param.data -= lr * param.grad.data

            print(total_loss_epoch[epoch])

        return total_loss_epoch


    def sgd_backward(self, criterion, n_epoch, lr):
        '''
        Compute the sgd algorithm
        inputs : neural net, loss_function, number of epochs, learning rate
        goal : get a minimum
        return : total_loss_epoch, grad_norm_epoch
        '''
        total_loss_epoch = [0 for i in range(n_epoch)]
        for epoch in range(n_epoch):
            print(epoch)

            # Compute full grad
            previous_net_grad = copy.deepcopy(self)  # update previous_net_grad
            previous_net_grad.zero_grad()
            total_loss_epoch[epoch], _ = \
                previous_net_grad.calculate_loss_grad(data_set, criterion, CLASSES)

            random.shuffle(index)
            for data_label in index:
                inputs, labels = data_set[data_label]
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                self.zero_grad()
                loss = self.partial_grad(inputs, labels, criterion)
                for para in self.parameters():
                    para.data -= lr * para.grad.data

        return total_loss_epoch


net_newalg = Net().to(DEVICE)
net_sgd = Net().to(DEVICE)

criterion = nn.CrossEntropyLoss()
start = time.time()
n_samples_total = len(train_loader)
n_samples = len(train_loader_list[0])

print('New Algorithm')
total_loss_epoch_newalg = \
    net_newalg.newalg_backward(criterion, EPOCHS, Learning_Rate_New)

print('SGD')
total_loss_epoch_sgd = \
    net_sgd.sgd_backward(criterion, EPOCHS, Learning_Rate_SGD)


end = time.time()
print('time is : ', end - start)
print('Finished Training')

print(total_loss_epoch_newalg)
print(total_loss_epoch_sgd)

plt.plot(range(EPOCHS), total_loss_epoch_newalg, lw=4, label="New Algorithm")
plt.plot(range(EPOCHS), total_loss_epoch_sgd, lw=4, label="SGD")

plt.yscale('log')
plt.xlabel('iteration')
plt.ylabel('loss_function')
plt.title('objective functions evolution(lr=0.01)')
plt.legend()
# plt.savefig('lr001.png')
plt.show()