from torchvision import datasets, transforms
import torch
import numpy as np
import random
import warnings
import time

import config
from classifier.network.training import train_network
from classifier.network.training import vectorized_train_network
from classifier import get_data
from gui import window

from classifier.network.functions import hypothesis
from classifier.network.functions import cost
from classifier.network.functions import reform
from classifier.network.functions import to_binary

np.set_printoptions(edgeitems=30, linewidth=1000,
    formatter=dict(float=lambda x: "%.3g" % x))
np.set_printoptions(suppress=True)

training_data, training_labels = get_data.get_training_data(config.samples)
training_data[training_data > 0] = 1

training_data = np.reshape(training_data, (config.samples, 784))
training_labels_binary = to_binary.to_binary(training_labels)


cv_data, cv_labels = get_data.get_testing_data(config.cv_samples)
cv_data[cv_data > 0] = 1
cv_data = np.reshape(cv_data, (config.cv_samples, 784))


if config.training:
    print("beginning training:")
    print("using random theta: {}".format(config.random_theta))
    print("training network size: {}".format(config.network_size))
    print("using splices: {}".format(config.split_data))
    if config.split_data:
        print("splice size: {}".format(config.samples/config.splices))
    vectorized_start = time.time()
    vectorized_train_network.train_network(training_data, training_labels_binary)
    vectorized_end = time.time()
    print("training finished, took: {} seconds".format(vectorized_end-vectorized_start))

if config.test_cv:
    print("beginning cross validation")

    thetas = reform.reform_theta(np.genfromtxt(config.theta_dir))

    correct = 0
    for i in range(config.cv_samples):
        guess = np.argmax(hypothesis.hypothesis(thetas, cv_data[i]))
        # print("cv data: {}".format(i))
        # print("guess: {}".format(guess))
        # print("acutal: {}".format(cv_labels[i]))
        # print("visual representation: {}".format(cv_data[i]))

        if guess == cv_labels[i]:
            correct += 1

    print("finished cross validation")
    print("accuracy: {} out of {}".format(correct, config.cv_samples))

if config.test_train:
    print("beginning testing trained data")
    correct = 0
    for i in range(config.samples):
        h = hypothesis.hypothesis(thetas, training_data[i])
        # print("training_data[{}]: {}".format(i,np.reshape(training_data[i],(28,28))))
        # print("training_label[{}]: {}".format(i,training_labels[i]))
        guess = np.argmax(h)
        # print("h: {}".format(h))
        # print("guess: {}".format(guess))
        if guess == training_labels[i]:
            correct += 1

    print("finished testing trained data")
    print("accuracy: {} out of {}".format(correct, config.samples))

if config.start_gui:
    print("starting gui")

    window.start_gui()
