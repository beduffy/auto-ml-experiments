# Adapted from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/logistic_regression/main.py#L35-L42

import sys

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  # todo look up data loader again?
                                          batch_size=batch_size,
                                          shuffle=False)


# Model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# model = LogisticRegression(input_size, num_classes)
model = CNN().cuda()

contains_Conv2D = False
for idx, module in enumerate(model.modules()):
    if type(module) == nn.Conv2d:
        contains_Conv2D = True
    print(module, 'type(m) == {}'.format(type(module)))



# Loss and Optimizer
# Softmax is internally computed.
# Set parameters to be updated.
criterion = nn.CrossEntropyLoss()  # todo work out how to calculate crossentropy using documentation. I couldn't before
# todo study softmax and cross entropy more
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training the Model
for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0
    for i, (images, labels) in enumerate(train_loader):
        if contains_Conv2D:
            images = Variable(images).cuda()
        else:
            images = Variable(images.view(-1, 28 * 28))

        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Record the correct predictions for training data
        train_total += labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels.data).sum()

        if (i + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
                     loss.data[0]))  # 600 steps of 100 batch size == 60000

    print('Train accuracy: {}'.format((100 * train_correct / train_total)))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    if contains_Conv2D:
        images = Variable(images).cuda()
    else:
        images = Variable(images.view(-1, 28 * 28))
    # labels.cuda()
    outputs = model(images).cpu()
    _, predicted = torch.max(outputs.data, 1)  # 2nd output is index. max_indices AKA argmax
    total += labels.size(0)  # labels size is 100 == batch size
    correct += (predicted == labels).sum()  # each size (100, 1). returns boolean array. Can then sum.

print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Model
torch.save(model.state_dict(), 'model.pkl')
