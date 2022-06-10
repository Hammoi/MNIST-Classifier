import numpy as np
import config

def to_binary(y):

    yZeros = np.zeros((np.shape(y)[0], config.output))
    yZeros[np.arange(np.size(y)),y.astype(int)] = 1
    y = yZeros

    return y
