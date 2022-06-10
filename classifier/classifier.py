import numpy as np

from classifier.network.training import train_network

from classifier.network import get_data

x = get_data.get_training_data()
y = get_data.get_training_label()
print(np.shape(x))
print(np.shape(y))
print(x[0])
print(y[0])
