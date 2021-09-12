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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
ids = [0, 1]
torch.manual_seed(0)

# 定义一些超参数
BATCH_SIZE = 1024  # batch_size即每批训练的样本数量
EPOCHS = 20  # 循环次数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，即device定义为CUDA或CPU
CLASSES = 10
BETA = 0.9
Learning_Rate_New = 0.001

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='data', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)


# -------------标签为data_label的train_data----------------------------------

train_loader_list = []
for data_label in range(CLASSES):
    indices = [1 if train_set[i][1] == data_label else 0 for i in range(len(train_set))]
    indices = torch.tensor(indices)
    indices = torch.nonzero(indices)
    Sampler = sampler.SubsetRandomSampler(indices=indices)
    train_loader_part = torch.utils.data.DataLoader(train_set,
                                                    batch_size=BATCH_SIZE,
                                                    sampler=Sampler,
                                                    shuffle=False)
    train_loader_list.append(train_loader_part)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Function to predict data's classification
        args : neural net, data
        output : prediction
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

    def calculate_loss_grad(self, dataset, criterion):
        """
        Function to compute the full loss and the full gradient
        args : dataset, loss function and number of samples
        return : total loss and full grad norm
        """
        total_loss = 0.0
        full_grad = []
        for data_label in range(CLASSES):
            inputs, labels = dataset[data_label]
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            total_loss += self.partial_grad(inputs, labels, criterion).data

        for param in self.parameters():
            full_grad.append(param.grad)

        return total_loss, full_grad

    def cal_inner_product(self, grad, param):
        inner_product = 0.0
        norm_grad = 0.0
        for element1, element2 in zip(grad, param):
            grad1_np = element1.data
            grad2_np = element2.grad.data
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
        # total_loss_epoch = [0 for i in range(n_epoch)]
        total_loss_epoch = numpy.zeros(n_epoch)
        index = [i for i in range(CLASSES)]
        self.zero_grad()

        data_set = []
        for data_label in range(CLASSES):
            data_set.append(next(iter(train_loader_list[data_label])))

        for epoch in range(n_epoch):
            print(epoch)

            # initialize iteration beginning
            previous_net_grad = copy.deepcopy(self)  # update previous_net_grad

            # Compute full grad
            previous_net_grad.zero_grad()
            total_loss_epoch[epoch], full_grad = previous_net_grad.calculate_loss_grad(data_set, criterion)

            # Run over the dataset
            random.shuffle(index)

            for grad, param in zip(full_grad, self.parameters()):
                # param.grad.data -= inner_product * grad.data
                param.data -= lr * grad.data
            # for data_label in index:
            #     inputs, labels = data_set[data_label]
            #     inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            #
            #     # Compute cur stoc grad
            #     self.zero_grad()
            #     loss = self.partial_grad(inputs, labels, criterion)
            #
            #     inner_product = self.cal_inner_product(full_grad, self.parameters())
            #
            #     for grad, param in zip(full_grad, self.parameters()):
            #         param.grad.data -= inner_product * grad.data
            #         param.data -= lr * param.grad.data
            print(total_loss_epoch[epoch])

        return total_loss_epoch

net_newalg = Net().to(DEVICE)

criterion = nn.CrossEntropyLoss()
start = time.time()
n_samples_total = len(train_loader)
n_samples = len(train_loader_list[0])
print([n_samples, n_samples_total])

print('New Algorithm')
total_loss_epoch_newalg = \
    net_newalg.newalg_backward(criterion, EPOCHS, Learning_Rate_New)

end = time.time()
print('time is : ', end - start)
print('Finished Training')

print(total_loss_epoch_newalg)