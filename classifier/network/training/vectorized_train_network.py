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

    g = [] #gradient
    for i in range(len(t)):
        g.append(np.zeros(np.shape(t[i])))

    for j in range(0,config.iterations):

        iteration_start = time.time()
        #print("training data {}".format(i))
        c = 0
        costs = []
        gradients = []
        if config.split_data:
            splice_size = int(config.samples/config.splices)
            for i in range(config.splices):
                x_splice = x[i*splice_size:(i+1)*(splice_size)]
                y_splice = y[i*splice_size:(i+1)*(splice_size)]
                hypothesis0 = hypothesis.hypothesis(t, x_splice)
                gradient = computeGradient.computeGradient(t, x_splice, y_splice)
                gradients.append(gradient)

                cost0 = cost.cost(hypothesis0,y_splice)
                costs.append(cost0)

            c = sum(costs)/len(costs)

            gradient_sum = []
            for i in range(len(gradients[0])):
                gradient_sum.append(sum([item[i] for item in gradients]))
            g = []
            for i in range(len(gradient_sum)):
                g.append(gradient_sum[i]/len(gradients))

        else:
            hypothesis0 = hypothesis.hypothesis(t, x)
            gradient = computeGradient.computeGradient(t, x, y)


            cost0 = cost.cost(hypothesis0,y)

            g = gradient
            c = cost0


        #either place epsilon here or place at t - g

        if(config.split_data):
            costs = []
            gradients = []

        for i in range(len(t)):
            temp = np.copy(t[i])
            temp[:,0] = 0
            t[i] = t[i]-(g[i] + config.lamb*temp/m) #adds regularization

        iteration_end = time.time()
        print("finished iteration {} of {}. cost: {}. computation time: {} seconds".format(j+1, config.iterations, c, iteration_end-iteration_start))

    #print("finished, saving thetas")

    flatThetas = np.array(())
    for i in range(len(t)):
        flatThetas = np.append(flatThetas, t[i])

    np.savetxt(config.theta_dir, flatThetas) #Overwrites current theta values. TODO: fix numpy conversion


    #print("overall cost improvement: {}%".format(100*abs(costs[0]-costs[-1])/((costs[0]+costs[-1])/2)))
