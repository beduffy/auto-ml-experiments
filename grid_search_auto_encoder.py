import sys
from random import shuffle

import numpy as np
from sklearn import datasets
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# Hyperparameters
# num_epochs = 50
# lr = 0.01

def get_iris():
    iris = datasets.load_iris()

    X = iris.data[:, 0:4]
    # y = iris.target

    batch_size = 30

    return X, batch_size

def get_boston():
    boston = datasets.load_boston()
    X = boston.data

    batch_size = 30

    return X, batch_size

def get_toy_simple_dataset(num_in_range=10, num_rows=1000, shuffle_range=False):
    X = []
    for i in range(num_rows):
        x = list(range(num_in_range))
        if shuffle_range:
            shuffle(x)
        X.append(x)

    X = np.array(X)
    batch_size = 10
    return X, batch_size

# X, batch_size = get_iris()
# X, batch_size = get_boston()
# X, batch_size = get_toy_simple_dataset()
# X, batch_size = get_toy_simple_dataset(num_rows=10000, shuffle_range=True)
X, batch_size = get_toy_simple_dataset(num_in_range=4, num_rows=10000, shuffle_range=True)
# X, batch_size = get_toy_simple_dataset(num_in_range=8, num_rows=1000, shuffle_range=False)

# todo possibly normalise
print(X)
print(X.shape)
train_loader = torch.utils.data.DataLoader(dataset=X, batch_size=batch_size, shuffle=True)

def get_layer_sizes_for_autoencoder(input_size=10, num_layers_in_half=2):
    layer_sizes = []

    size = input_size
    for i in range(num_layers_in_half):
        layer_sizes.append(size // 2)
        size = size // 2

    layer_sizes += layer_sizes[:-1][::-1]
    layer_sizes.insert(0, input_size)
    layer_sizes.append(input_size)

    return layer_sizes

# for i in range(4, 20):
#     print(i, get_layer_sizes_for_autoencoder(i))
# sys.exit()

# class AutoEncoder(nn.Module):
#     def __init__(self, input_and_output_size, hidden_size):
#         super(AutoEncoder, self).__init__()
#         self.fc1 = nn.Linear(input_and_output_size, hidden_size)
#         self.activation_func = nn.Sigmoid() #nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, input_and_output_size)
#
#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.activation_func(out)
#         out = self.fc2(out)
#         return out

# class AutoEncoder(nn.Module):
#     def __init__(self, input_and_output_size, hidden_size):
#         super(AutoEncoder, self).__init__()
#         self.fc1_en = nn.Linear(input_and_output_size, 7)
#         self.fc2_en = nn.Linear(7, 3)
#         self.activation_func = nn.Sigmoid() #nn.ReLU()
#         self.fc1_de = nn.Linear(3, 7)
#         self.fc2_de = nn.Linear(7, input_and_output_size)
#
#     def forward(self, x):
#         out = self.fc1_en(x)
#         out = self.activation_func(out)
#         out = self.fc2_en(out)
#         out = self.activation_func(out)
#         out = self.fc1_de(out)
#         out = self.activation_func(out)
#         out = self.fc2_de(out)
#         return out

class AutoEncoder(nn.Module):
    def __init__(self, layer_sizes):
        super(AutoEncoder, self).__init__()

        #[10, 7, 3, 7, 10]
        bottle_neck_layer_size = min(layer_sizes)

        layers = []
        for i in range(len(layer_sizes) - 1):
            input_size, output_size = layer_sizes[i], layer_sizes[i+1]
            layers.append(nn.Linear(input_size, output_size))
            if i != len(layer_sizes) - 2:
                layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out

def weight_init(m):
    # print(m)
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, std=0.01)
        # print(m.weight)

def find_best_initial_learning_rate(ae):
    best_lr = -100
    lowest_loss = 100000000
    num_epochs_for_trying_lr = 10

    learning_rates_to_try = [1.0, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.0000001]

    # strange bug here with beginning loss always the same?

    for lr in learning_rates_to_try:
        ae.apply(weight_init)
        print(ae.layers[0].weight.data)
        end_loss = train_autoencoder(ae, lr, num_epochs_for_trying_lr, True)
        print(lr, end_loss)

        if end_loss < lowest_loss:
            best_lr, lowest_loss = lr, end_loss
            print('New best:', best_lr, lowest_loss)

    print('\nFinished hyperparameter optimisation. Best Learning Rate: {}, lowest loss: {} after {} epochs\n'.format(best_lr, lowest_loss, num_epochs_for_trying_lr))
    return best_lr

def train_autoencoder(ae, lr, num_epochs=50, to_print=True):
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(ae.parameters(), lr=lr, nesterov=True, momentum=0.9, dampening=0)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training the Model
    for epoch in range(num_epochs):
        scheduler.step()

        total_loss = 0.0

        for i, iris_data_it in enumerate(train_loader):
            # print(iris_data_it)
            iris_data_it = Variable(iris_data_it.float())

            # # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = ae(iris_data_it)
            loss = criterion(outputs, iris_data_it)
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]
        if to_print and ((epoch + 1) <= 5 or epoch > (num_epochs - 1) - 5):
            print('Epoch: [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, total_loss))

    if to_print:
        print('Test one example: ')
        print(X[0])
        output = ae(Variable(torch.FloatTensor(X[0])))
        print(output.data.cpu().numpy())

    return total_loss

# middle_layer_size = int(X.shape[1] / 2)
# ae = AutoEncoder(X.shape[1], middle_layer_size)
# print(ae)

layer_sizes = get_layer_sizes_for_autoencoder(X.shape[1], 2)
ae1 = AutoEncoder(layer_sizes)
print(ae1)

best_lr = find_best_initial_learning_rate(ae1)

print('Beginning to train autoencoder with Learning rate: {}'.format(best_lr))
train_autoencoder(ae1, best_lr)
