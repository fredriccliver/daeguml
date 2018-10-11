### https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/


#%% DEFINE FUNCTIONS
# Backprop on the Seeds Dataset
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
import inspect
import sys
from pprint import pprint

custom_hist = []

# Load a CSV file
def load_csv(filename):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Find the min and max values for each column
def dataset_minmax(dataset):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Calculate neuron activation for an input
def activate(weights, inputs):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
        # print("activation", activation)
    return activation

# Transfer neuron activation
def transfer(activation):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    
    ## sigmoid
    # return 1.0 / (1.0 + exp(-activation))
    
    ## ReLu
    # return max(0, activation)

    ## Elu
    if(activation>=0):
        return activation
    else:
        return (exp(activation)-1)



# Forward propagate input to a network output
def forward_propagate(network, row):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    
    ## sigmoid
    # return output * (1.0 - output)
    
    ## ReLu
    # if(output > 0):
    #     return 1
    # else:
    #     return 0

    ## Elu
    if(output > 0):
        return 1
    else:
        return (exp(output)-1)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])
            custom_hist.append(neuron['delta'])
        

# Update network weights with error
def update_weights(network, row, l_rate):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']
            

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Make a prediction with a network
def predict(network, row):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    # print("[FUNCTION] : ",inspect.getframeinfo(inspect.currentframe()).function)
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)

print("DEFINED")


#%% FIT PREDICT EXAMPLE
custom_hist = []
# Test Backprop on Seeds dataset
seed(1)
# load and prepare data
filename = 'seeds_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 2         # how many seperates data for validation
l_rate = 0.5        # learning rate
n_epoch = 30        # learn repeat
n_hidden = 10       # neuron count (hidden)
back_propagation_hist = []
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


#%%
import matplotlib.pyplot as plt
plt.plot(custom_hist)
# pprint(custom_hist)


#%% make hyperparameter history
l_rate = .0001
hyperparam_hist = []
for n_epoch in (5, 10, 15, 20, 25, 30, 35):
    for n_hidden in (5, 10, 15, 20, 25, 30, 35):
        scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
        hyperparam_hist.append([n_epoch, n_hidden, (sum(scores)/float(len(scores)))])
        print("n_epoch, n_hidden ", n_epoch, n_hidden)

print("hyperparam_hist : ", hyperparam_hist)


#%% hyperparameter-accuracy plotting
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig = plt.figure()
title = "l_rate:"+str(l_rate)
fig.suptitle(title)

ax = fig.add_subplot(111, projection='3d')
for i in range(0, len(hyperparam_hist)):
    ax.scatter(xs=hyperparam_hist[i][0], ys=hyperparam_hist[i][1], zs=hyperparam_hist[i][2], c='r', marker="o")
ax.set_xlabel('n_epoch')
ax.set_ylabel('n_hidden')
ax.set_zlabel('accuracy')
plt.show()

fig = plt.figure()
bx = fig.gca(projection='3d')
df = pd.DataFrame(hyperparam_hist, columns=['x','y','z'])
surf = bx.plot_trisurf(df.x, df.y, df.z, linewidth=0.1)
bx.set_xlabel('n_epoch')
bx.set_ylabel('n_hidden')
bx.set_zlabel('accuracy')
plt.show()

fig = plt.figure()
cx = fig.gca(projection='3d')
df = pd.DataFrame(hyperparam_hist, columns=['x','y','z'])
surf = cx.plot_trisurf(df.x, df.y, df.z, linewidth=0.1)
cx.set_xlabel('n_epoch')
cx.set_ylabel('n_hidden')
cx.set_zlabel('accuracy')
cx.view_init(azim=15)
plt.show()

fig = plt.figure()
cx = fig.gca(projection='3d')
df = pd.DataFrame(hyperparam_hist, columns=['x','y','z'])
surf = cx.plot_trisurf(df.x, df.y, df.z, linewidth=0.1)
cx.set_xlabel('n_epoch')
cx.set_ylabel('n_hidden')
cx.set_zlabel('accuracy')
cx.view_init(azim=-15)
plt.show()


