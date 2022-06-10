import numpy as np

from classifier.network.functions import hypothesis
from classifier.network.functions import reform
import config

def sigmoid(z):
    ## print(-z)
    return 1.0 / (1.0 + np.exp(-z.astype(np.longdouble)))
def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z));

def computeGradient(t, x, y, flattenGradient=False): #For only one input and output right now

    # if isinstance(t,np.ndarray) and np.size(np.shape(t)) == 1 and not isinstance(t,list):
    #     # print("gradient yes")
    #     t = reform.reform_theta(t) #for bfgs, whose theta input is flattened
    # print("computing gradient")
    h,a,z = hypothesis.hypothesis(t, x, computingGradient=True) #Step 1
    # print("\n\n\ni love you u big big dumbie and since u are a dum dum, u prob don't know how to calculate my love since it grows exponentially and i love u in every universe")
    # print("mwah mwah mwah mwah mwah mwah mwah (7 mwahs for good luck)")
    # print("hai baby, rmbr that i love u forever and always in all ways")
    # print("okie u can continue to do hard stuff that i don't understand and because u are a dum dumbie")
    # print("p.s u should program me a sleep tracker (a cute one)")
    # print("another p.s u should marry me in the future")
    # print("another another p.s we stay together forever and i need u cuz u are my happy ball of dumbie \n\n\n")

    # print("h: {}".format(h))
    # print("h shape: {}\n\n".format(np.shape(h)))
    # print("a: {} \n\n {} \n\n {} \n\n {} \n\n {} \n\n".format(a[0],a[1],a[2],a[3],a[4]))
    # print("a shape: {}, {}, {}, {}, {}\n\n".format(np.shape(a[0]),np.shape(a[1]),np.shape(a[2]),np.shape(a[3]),np.shape(a[4])))
    # print("z: {} \n\n {} \n\n {} \n\n {} \n\n".format(z[0],z[1],z[2],z[3]))
    # print("z shape: {}, {}, {}, {}\n\n".format(np.shape(z[0]),np.shape(z[1]),np.shape(z[2]),np.shape(z[3])))
    # print("y: {}".format(y))
    # print("y shape: {}".format(np.shape(y)))

    deltaList = []


    #This goes backwards (back-propagation)
    delta = a[-1]-y #for output units
    deltaList.insert(0, delta) #delta 4

    for i in range(config.network_size[1]):
        j = config.network_size[1]-i
        # print("\n\nz[{}]: {}".format(j-1,z[j-1]))
        if np.size(np.shape(z[j-1])) == 1:
            # print("called here")
            z[j-1] = np.insert(z[j-1], 0, 1, axis=0) #add 1 to account for bias unit
        else:
            # print("called there")
            z[j-1] = z[j-1].T
            z[j-1] = np.insert(z[j-1], 0, 1, axis=1)
        # print("z[{}]: {}\n\n".format(j-1,z[j-1]))

        delta = np.dot(t[j].T,delta.T).T
        # print("delta: {}".format(delta.T))
        # print("sigmoidGradient: {}".format(sigmoidGradient(z[j-1])))
        delta = delta * sigmoidGradient(z[j-1])
        # print("delta: {}".format(delta))
        if np.size(np.shape(delta)) == 1:
            delta = np.delete(delta, 0) #remove bias unit's delta
            deltaList.insert(0, delta) #delta 3
        else:
            delta = np.delete(delta, 0, axis=1) #remove bias unit's delta
            # print("delta after delete: {}".format(delta))
            deltaList.insert(0, delta) #delta 3


    gradientList = []
    for i in range(config.network_size[1]+1):
        # print("deltaList[{}]: {}".format(i,deltaList[i]))
        # print("a[{}]: {}".format(i,np.array([a[i]])))
        if np.size(np.shape(deltaList[0])) == 1:
            gradient = np.dot(np.array([deltaList[i]]).T,np.array([a[i]]))

        else:
            gradient = deltaList[i][...,None]*a[i][:,None,:]
            # print("gradient: {}".format(gradient))
            gradient = np.sum(gradient, axis=0)
            gradient = gradient/np.shape(x)[0]
            # print("gradient: {}\n\n\n".format(gradient))
        gradientList.append(gradient)


    if flattenGradient:
        flatGradient = np.array(())
        for i in range(config.network_size[1]+1):
            flatGradient = np.concatenate((flatGradient.flatten(),gradientList[i].flatten()))
        return flatGradient
    else:
        return gradientList
