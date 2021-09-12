import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import os
import torch.utils.data.sampler as sampler
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,5"
ids = [0, 1]
torch.manual_seed(0)

# 定义一些超参数
BATCH_SIZE = 32  # batch_size即每批训练的样本数量
EPOCHS = 20  # 循环次数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，即device定义为CUDA或CPU
Learning_Rate_SVRG = 0.01

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='data', train=True, download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

test_set = torchvision.datasets.CIFAR10(root='data', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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

    def calculate_loss_grad(self, dataset, criterion, n_samples):
        """
        Function to compute the full loss and the full gradient
        args : dataset, loss function and number of samples
        return : total loss and full grad norm
        """
        total_loss = 0.0
        full_grad = []
        for i_grad, (inputs, labels) in enumerate(dataset):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            total_loss += (1./n_samples) * self.partial_grad(inputs, labels, criterion).data

        for param in self.parameters():
            full_grad.append((1. / n_samples) * param.grad)

        return total_loss, full_grad

    def svrg_backward(self, criterion, n_epoch, lr):
        """
        Function to update weights with a SVRG backpropagation
        args : loss function, number of epochs, learning rate
        return : total_loss_epoch, grad_norm_epoch
        """
        total_loss_epoch = [0 for i in range(n_epoch)]
        test_accuracy_epoch = [0 for i in range(n_epoch)]

        for epoch in range(n_epoch):
            print(epoch)
            previous_net_sgd = copy.deepcopy(self) # update previous_net_sgd
            previous_net_grad = copy.deepcopy(self) # update previous_net_grad

            # Compute full grad
            previous_net_grad.zero_grad()
            total_loss_epoch[epoch], _ = \
                previous_net_grad.calculate_loss_grad(train_loader, criterion, n_samples_total)

            for i_data, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # Compute prev stoc grad
                previous_net_sgd.zero_grad()
                loss = previous_net_sgd.partial_grad(inputs, labels, criterion)

                # Compute cur stoc grad
                self.zero_grad()
                loss = self.partial_grad(inputs, labels, criterion)

                # Backward
                for param1, param2, param3 in zip(self.parameters(), previous_net_sgd.parameters(),
                                                  previous_net_grad.parameters()):
                    param1.data -= (lr) * (param1.grad.data - param2.grad.data + (1. / n_samples_total) * param3.grad.data)

            test_accuracy_epoch[epoch] = self.test("SVRG")
            print(total_loss_epoch[epoch], test_accuracy_epoch[epoch])

        return total_loss_epoch, test_accuracy_epoch

    def test(self, name):
        correct = 0
        total = 0
        for (images, labels) in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = self.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Accuracy of the %s on the 10000 test images: %d %%' % (name, 100 * correct / total))

        return 100.0 * correct / total


net_svrg = Net().to(DEVICE)
criterion = nn.CrossEntropyLoss()
start = time.time()
n_samples_total = len(train_loader)


print('SVRG')
total_loss_epoch_svrg, test_accuracy_epoch_svrg = \
    net_svrg.svrg_backward(criterion, EPOCHS, Learning_Rate_SVRG)

end = time.time()
print('time is : ', end - start)
print('Finished Training')

print(total_loss_epoch_svrg)
print(test_accuracy_epoch_svrg)