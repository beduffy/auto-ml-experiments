import random

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Genetic Hyper Parameters
num_times_breed = 5
population_size = 10
breeding_population_fraction = 0.5
breeding_population = int(population_size * breeding_population_fraction)

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def create_net():
    net = Net(input_size, hidden_size, num_classes)
    #todo randomize weights with better init? or not.
    net.cuda()
    return net

def test_model(net):
    # Test the Model
    correct = 0
    total = 0
    total_loss = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28 * 28)).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
        labels = Variable(labels.cuda())
        loss = criterion(outputs, labels)
        total_loss += loss

    accuracy = 100 * correct / total
    # print('Accuracy of the network on the 10000 test images: %d %%\nLoss: %f' % (accuracy, total_loss))

    return total_loss, accuracy

def mutate_two_models(model1, model2):
    mutated_net = Net(input_size, hidden_size, num_classes)


    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()
    # mutated_net.load_state_dict({name: (model1_state_dict[name] + model2_state_dict[name]) / 2
    #                        for name in model1_state_dict})

    # mutated_net.load_state_dict({name: model1_state_dict[name] + (torch.randn(model1_state_dict[name].size()) * 0.1).cuda()
    #                              for name in model1_state_dict})  # best so far

    new_model_state_dict = {}

    for name in model1_state_dict:
        rnd_scalar = random.uniform(-1, 1)
        new_model_state_dict[name] = model1_state_dict[name] + (torch.randn(model1_state_dict[name].size()) * rnd_scalar).cuda()

    mutated_net.load_state_dict(new_model_state_dict)

    # print(model1_state_dict['fc1.weight'])
    # print(model2_state_dict['fc1.weight'])
    # print(mutated_net.state_dict()['fc1.weight'])

    mutated_net.cuda()
    return mutated_net

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# create initial population of models
all_models = []
for i in range(population_size):
    all_models.append(create_net())

for round in range(100000):
    print('Iteration: {}'.format(round))

    best_model_idx_loss = []
    for idx, model in enumerate(all_models):
        loss, accuracy = test_model(model)
        best_model_idx_loss.append((idx, loss.data[0], accuracy))

    # best_model_idx_loss_sorted = sorted(best_model_idx_loss, key=lambda x: x[1]) # sort by loss. Does badly for some reason?
    best_model_idx_loss_sorted = sorted(best_model_idx_loss, key=lambda x: x[2], reverse=True) # sort by accuracy
    all_accuracies = [x[2] for x in best_model_idx_loss_sorted]
    all_losses = [x[1] for x in best_model_idx_loss_sorted]
    all_indices_sorted = [x[0] for x in best_model_idx_loss_sorted]
    lowest_accuracy = min(all_accuracies)
    highest_accuracy = max(all_accuracies)
    lowest_loss = min(all_losses)

    top_model_indices_and_losses = best_model_idx_loss_sorted[:breeding_population]
    top_model_indices = [x[0] for x in top_model_indices_and_losses]
    worst_model_indices_and_losses = best_model_idx_loss_sorted[-breeding_population:]
    worst_model_indices = [x[0] for x in worst_model_indices_and_losses]

    print(best_model_idx_loss_sorted)
    # print(top_model_indices_and_losses)
    # print(top_model_indices)
    # print(worst_model_indices)

    # Remove models outside breeding population
    all_models = [all_models[idx] for idx in top_model_indices]
    # top_models = [model for idx, model in enumerate(all_models) if idx in top_model_indices]
    # top_models = [all_models[idx] for idx in top_model_indices]
    # all_models = all_models[:breeding_population]
    # print(top_models)


    for i in range(num_times_breed):
        # two_random_models = [top_models[i] for i in sorted(random.sample(range(len(top_models)), 2))]
        two_random_models = [all_models[i] for i in sorted(random.sample(range(len(all_models)), 2))]
        #assert two_random_models[0] != two_random_models[1]
        new_model = mutate_two_models(two_random_models[0], two_random_models[1])
        all_models.append(new_model)

    # for w_m in worst_model_indices:
    #     del all_models[w_m]

    print('Num models: {}. Best acc: {}. Worst acc: {}. Lowest loss: {}'.format(len(all_models), highest_accuracy, lowest_accuracy, lowest_loss))
