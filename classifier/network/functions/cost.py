import numpy as np
import math
def cost(h,y): #where h and y are same size and numpy arrays, NOT REGULARIZED
    #Encountering nan/inf values are no problem, negligable (nan=0, inf=1)
    if np.size(np.shape(y)) == 1:
        m = np.size(y)
    else:
        m = np.shape(y)[0]

    cost = (-1/m) * (y*np.log(h) + (1-y)*np.log(1-h))

    cost_sum = np.sum(cost)
    # if math.isnan(cost_sum):
        #print("nan cost encountered, printing all values:")
    # print("\n\n\nh: {}\n".format(h))
    # print("y: {}\n".format(y))
    # print("cost: {}\n".format(cost))
    # print("cost_sum: {}\n\n\n".format(cost_sum))
    return np.sum(cost)
