import numpy as np
import config

def reform_theta(t): #takes flattened thetas
    thetaList = []
    print("Reforming theta")
    for i in range(0,config.network_size[1]+1):
        if i == 0: #input layer
            numberOfThetas = (config.input+1)*(config.network_size[0])
            theta = np.reshape(t[0:numberOfThetas], (config.network_size[0], config.input+1))
        elif i == config.network_size[1]: #output layer
            numberOfThetas = config.output*(config.network_size[0]+1) #should be the same value
            theta = np.reshape(t,(config.output,config.network_size[0]+1))
        else: #hidden layers
            numberOfThetas = config.network_size[0]*(config.network_size[0]+1)
            theta = np.reshape(t[0:numberOfThetas], (config.network_size[0],config.network_size[0]+1))

        t = t[numberOfThetas:]
        thetaList.append(theta)

    return thetaList
