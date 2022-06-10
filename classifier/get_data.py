from torchvision import datasets, transforms
import numpy as np

import random

import config


training_data = np.array([])
def get_training_data(number_of_samples):
    global training_data
    if np.size(training_data) == 0:
        train_set = datasets.MNIST('./data', train=True, download=True)
        training_data = train_set.data.numpy()
        training_labels = train_set.targets.numpy()

    start_index = random.randint(0, config.available_samples-number_of_samples)
    print("pulling training data starting at index: {}".format(start_index))
    return training_data[start_index:start_index+number_of_samples], training_labels[start_index:start_index+number_of_samples]

testing_data = np.array([])
def get_testing_data(number_of_samples):
    global testing_data
    if np.size(testing_data) == 0:
        test_set = datasets.MNIST('./data', train=False, download=True)
        testing_data = test_set.data.numpy()
        testing_labels = test_set.targets.numpy()

    start_index = random.randint(0, config.available_cv_samples-number_of_samples)
    print("pulling cv data starting at index: {}".format(start_index))

    return testing_data[start_index:start_index+number_of_samples], testing_labels[start_index:start_index+number_of_samples]
