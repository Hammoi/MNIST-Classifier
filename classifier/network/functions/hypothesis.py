import numpy as np

from classifier.network.functions import reform
import config

def sigmoid(z):
    #print(-z)
    return 1.0 / (1.0 + np.exp(-z.astype(np.longdouble)))

def hypothesis(t, x, placeholder=0, computingGradient = False, printData = False):

    if isinstance(t,np.ndarray) and np.shape(t)[0] == 1:
        print("yes")
        t = reform.reform_theta(t) #for bfgs, whose theta input is flattened

    # print("hypothesis data:")
    # print(np.shape(x))
    # print(np.shape(t))

    aList = []
    zList = []
    #print("computing hypothesis")
    #print("x shape: {}".format(np.size(np.shape(x))))
    #print("x: {}".format(x))
    if np.size(np.shape(x)) == 1:
        x = np.insert(x, 0, 1, axis=0) #Add bias unit to input
    else:
        x = np.insert(x, 0, 1, axis=1) #for input with multiple x values
    #print("x: {}".format(x))
    aList.append(x) #a0
    a = x
    for i in range(config.network_size[1]):
        #print("t[{}] shape: {}".format(i, np.shape(t[i])))
        #print("a shape: {}".format(np.shape(a.T)))
        z = np.dot(t[i], a.T)
        #print("z: {}".format(z))
        a = sigmoid(z).T
        #print("a: {}".format(a))
        #print("a shape: {}".format(np.shape(a)))
        if np.size(np.shape(a)) == 1:
            a = np.insert(a, 0, 1, axis=0) #Add bias unit to a
        else:
            #print("called")
            a = np.insert(a, 0, 1, axis=1) #Add bias unit to a for multiple x values
        #print("a: {}".format(a))
        #print("a shape: {}".format(np.shape(a)))

        if computingGradient:
            zList.append(z) #z0
            aList.append(a) #a1

    h = np.dot(t[config.network_size[1]], a.T)
    #print("h: {}".format(h))
    if computingGradient:
        zList.append(h) #z2

    h = sigmoid(h)

    if np.size(np.shape(h)) > 1:
        h = h.T
    #print("h: {}".format(h))
    if computingGradient:
        aList.append(h) #a3

    if computingGradient:
        return h, aList, zList
    else:
        return h
