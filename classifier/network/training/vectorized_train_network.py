import numpy as np
import time
import config

from classifier.network.functions import cost
from classifier.network.functions import hypothesis
from classifier.network.functions import computeGradient
from classifier.network.functions import reform

def train_network(x, y): #player = 0,1

    m = np.shape(x)[0]

    #[5,2]
    if config.random_theta:
        flat_thetas = np.random.uniform(-1,1,(config.network_size[0]*(config.input+1)
                                              + (config.network_size[1]-1)*(config.network_size[0]*(config.network_size[0]+1))
                                              + config.output*(config.network_size[0]+1))) #first layer (w/ input) + hidden layers + output layer
        t = reform.reform_theta(flat_thetas)
    else:
        t = reform.reform_theta(np.genfromtxt(config.theta_dir))

    costs = []
    c = 0 #cost
    g = [] #gradient
    for i in range(len(t)):
        g.append(np.zeros(np.shape(t[i])))

    for j in range(0,config.iterations):

        iteration_start = time.time()
        #print("training data {}".format(i))
        hypothesis0 = hypothesis.hypothesis(t, x)
        gradient = computeGradient.computeGradient(t, x, y)


        cost0 = cost.cost(hypothesis0,y)
        #either place epsilon here or place at t - g

        g = gradient


        costs.append(cost0)
        c = cost0

        for i in range(len(t)):
            temp = np.copy(t[i])
            temp[:,0] = 0
            t[i] = t[i]-(g[i] + config.lamb*temp/m) #adds regularization

        g = [] #gradient
        iteration_end = time.time()
        print("finished iteration {} of {}. cost: {}. computation time: {} seconds".format(j+1, config.iterations, c, iteration_end-iteration_start))

    #print("finished, saving thetas")

    flatThetas = np.array(())
    for i in range(len(t)):
        flatThetas = np.append(flatThetas, t[i])

    np.savetxt(config.theta_dir, flatThetas) #Overwrites current theta values. TODO: fix numpy conversion


    #print("overall cost improvement: {}%".format(100*abs(costs[0]-costs[-1])/((costs[0]+costs[-1])/2)))
