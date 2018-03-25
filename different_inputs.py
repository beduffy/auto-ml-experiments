import sys
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        self.seen_input_sizes = set()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(1, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        # input_dim = tuple(x.size())[1]
        # if input_dim not in self.seen_input_sizes:
        #     self.seen_input_sizes.add(input_dim)
        #     print('Added new tuple input size: {}'.format(tuple(x.size())))
        #     self.add_module('{}_fc1'.format(input_dim), nn.Linear(input_dim, self.hidden_size))
        #     # print(self)
        #
        # # out = self.named_parameters('{}_fc1'.format(tuple(x.size())))(x)
        # # out = self.named_modules('{}_fc1'.format(tuple(x.size())))(x)
        # for m in self.named_modules():
        #     # print(m)
        #     if m[0] in ['{}_fc1'.format(input_dim)]:
        #         out = m[1](x)
        #         break

        # sys.sysout.flush()
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = Net(30)

# seen_input_sizes = set()
# for i in range(100):
#     random_size_input = Variable(torch.rand(random.randint(1, 10), random.randint(1, 10)))
#     # print(random_size_input.size())
#     tuple_random_size_input = tuple(random_size_input.size())
#     # print(tuple_random_size_input)
#     # print(seen_input_sizes)
#     # if random_size_input.size() not in seen_input_sizes:
#     if tuple_random_size_input not in seen_input_sizes:
#         seen_input_sizes.add(tuple_random_size_input)
#     else:
#         print('Already in seen input sizes')


# def f1(x): return x
# def f2(x, x2): return x + x2
# def f3(x, x2, x3): return x + x2 + x3
# def f4(x, x2, x3, x4): return x + x2 + x3 + x4
# def f5(x, x2, x3, x4, x5): return x + x2 + x3 + x4 + x5
#
# all_functions = [f1, f2, f3, f4, f5]



def f(x):
    # return x[:, ]/
    return x.sum(axis=1)

N = 10
num_iterations = 100000
learning_rate = 0.0001

all_losses = []

# optimizer = torch.optim.Adam(net.parameters())
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

data = []
for idx, x_size in enumerate(range(1, 2)):
    # x_data = np.random.randn(N, x_size)
    x_data = np.random.randint(0, 100, size=(N, x_size))
    y_data = f(x_data)
    data.append((x_data, y_data))

for iteration in range(num_iterations):
    # for idx, x_size in enumerate(range(1, 6)):
    for x_data, y_data in data:
        x = Variable(torch.FloatTensor(x_data), requires_grad=False)
        y = Variable(torch.FloatTensor(y_data), requires_grad=False)

        # Forward + Backward + Optimize
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        optimizer.zero_grad()  # zero the gradient buffer
        y_pred = net(x)
        loss = (y_pred - y).pow(2).mean()
        print(loss.data[0])
        # print(x.data.numpy())
        # print(y.data.numpy())
        # print(x.data.numpy()[0])
        print('y: {}. y_pred: {}'.format(y.data.numpy()[0], y_pred.data.numpy()[0][0]))
        # print()

        loss.backward()
        optimizer.step()

        # print(net.state_dict())
        # a = net.state_dict()
        # print(net.state_dict()['1_fc1.weight'].grad)
        # print(net.state_dict()['1_fc1.weight'].data)
        # for name, parameter in net.named_parameters():
        #     if name in ['1_fc1.weight']:
        #         #grad_of_param[name] = parameter.grad
        #         print(parameter.grad)
        #         print(parameter.data)
        # sys.exit()


        all_losses.append(loss.data[0])

plt.plot(all_losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
