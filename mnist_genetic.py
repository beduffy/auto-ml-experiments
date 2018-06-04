import time
import threading
import datetime
import random

import matplotlib.pyplot as plt
from matplotlib.pyplot import ion, show, figure
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# ion() # enables interactive mode

# Hyper Parameters for model
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# Genetic Hyper Parameters
# num_times_breed = 5
save_figures = True
low = 1
high = 5

population_size = 25

replace_fraction = 0.3
breeding_population_fraction = 0.7
breeding_population = int(population_size * breeding_population_fraction)
num_times_breed = population_size - breeding_population
# num_times_to_breed_to_keep_same_population_size = breeding_population -

print('Hyperparams')
print('breeding_population: ', breeding_population)
print('num_times_breed: ', num_times_breed)

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
    # attempt at sexual reproduction
    # mutated_net.load_state_dict({name: (model1_state_dict[name] + model2_state_dict[name]) / 2
    #                        for name in model1_state_dict})

    # mutated_net.load_state_dict({name: model1_state_dict[name] + (torch.randn(model1_state_dict[name].size()) * 0.1).cuda()
    #                              for name in model1_state_dict})  # best so far

    new_model_state_dict = {}

    # asexual reproduction
    for name in model1_state_dict:
        # rnd_scalar = random.uniform(-1, 1)
        rnd_scalar = np.random.randint(1, 5)
        new_model_state_dict[name] = model1_state_dict[name] + (torch.randn(model1_state_dict[name].size()) * rnd_scalar).cuda()

    mutated_net.load_state_dict(new_model_state_dict)

    # print(model1_state_dict['fc1.weight'])
    # print(model2_state_dict['fc1.weight'])
    # print(mutated_net.state_dict()['fc1.weight'])

    mutated_net.cuda()
    return mutated_net

def mutate_model(model1, low=1, high=5):
    mutated_net = Net(input_size, hidden_size, num_classes)

    model1_state_dict = model1.state_dict()
    new_model_state_dict = {}

    # asexual reproduction
    for name in model1_state_dict:
        # rnd_scalar = np.random.randint(1, 5)
        rnd_scalar = np.random.uniform(low, high)
        new_model_state_dict[name] = model1_state_dict[name] + (torch.randn(model1_state_dict[name].size()) * rnd_scalar).cuda()

    mutated_net.load_state_dict(new_model_state_dict)

    # print(model1_state_dict['fc1.weight'])
    # print(model2_state_dict['fc1.weight'])
    # print(mutated_net.state_dict()['fc1.weight'])

    mutated_net.cuda()
    return mutated_net

def test_and_evaluate_model(model):
    loss, accuracy = test_model(model)
    best_model_idx_loss.append((idx, loss.data[0], accuracy))

if __name__ == "__main__":
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    counter_how_many_times_worst_than_previous = 0

    all_mean_accs = []
    all_worst_accs = []
    all_best_accs = []
    all_lowest_losses = []

    # create initial population of models
    all_models = []
    for i in range(population_size):
        all_models.append(create_net())

    for round in range(100000):
        start_time = time.time()
        print('Iteration: {}'.format(round))

        # todo multithread this loop? and join all after. Inference of model is the longer part.
        # todo is GIL blocking us?
        best_model_idx_loss = []
        threads = []
        for idx, model in enumerate(all_models):
            # t = threading.Thread(target=test_and_evaluate_model, args=(model, ))
            # t.daemon = True
            # threads.append(t)
            # t.start()

            # non-threading version
            loss, accuracy = test_model(model)
            best_model_idx_loss.append((idx, loss.data[0], accuracy))

        for t in threads:
            t.join()

        best_model_idx_loss_sorted = sorted(best_model_idx_loss, key=lambda x: x[1]) # sort by loss. Does badly for some reason? # todo doesn't change at all??
        # best_model_idx_loss_sorted = sorted(best_model_idx_loss, key=lambda x: x[2], reverse=True) # sort triple by accuracy
        all_indices_sorted = [x[0] for x in best_model_idx_loss_sorted]
        all_losses = [x[1] for x in best_model_idx_loss_sorted]
        all_accuracies = [x[2] for x in best_model_idx_loss_sorted]
        lowest_accuracy = min(all_accuracies)
        highest_accuracy = max(all_accuracies)
        mean_accuracy = np.mean(all_accuracies)
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


        # dynamically adjust how many you breed back in if worst accuracy is lower than last time
        extra_times_to_breed = 0
        if len(all_worst_accs) > 5 and lowest_accuracy < all_worst_accs[-1]:
            extra_times_to_breed = np.random.randint(1, 5)
            low += 0.1
            high += 0.1
            counter_how_many_times_worst_than_previous += 1
            print('low: {}, high: {}, counter_how_many_times_worst_than_previous: {}. Adding extra {} to population'.format(low, high, counter_how_many_times_worst_than_previous, extra_times_to_breed))
        else:
            counter_how_many_times_worst_than_previous = 0
            if len(all_models) - population_size > 10:
                extra_times_to_breed = -5


        for i in range(num_times_breed + extra_times_to_breed):
            random_best_model_idx = np.random.randint(0, 3)
            random_best_model = all_models[random_best_model_idx]

            # new_model = mutate_model(random_best_model)
            new_model = mutate_model(random_best_model)
            all_models.append(new_model)

        all_mean_accs.append(mean_accuracy)
        all_worst_accs.append(lowest_accuracy)
        all_best_accs.append(highest_accuracy)
        all_lowest_losses.append(lowest_loss)
        if save_figures and round > 1 and round % 2 == 0:
            plt.figure()
            plt.plot(all_best_accs, label='best accuracy')
            plt.plot(all_mean_accs, label='mean accuracy')
            plt.plot(all_worst_accs, label='worst accuracy')

            plt.legend()
            plt.xlabel('Number of iteration')
            plt.ylabel('Metric')
            plt.savefig('Figures/experiment-{}-round-{}'.format(datetime.datetime.now().strftime('%d-%m-%y'), round))

        print('Num models: {}. Best acc: {}.  Mean acc: {}. Worst acc: {}. Lowest loss: {}. Time taken: {:.2f}'.format(
            len(all_models),
            highest_accuracy,
            mean_accuracy,
            lowest_accuracy,
            lowest_loss,
            time.time() - start_time))
