from scipy.optimize import minimize
import numpy as np

from classifier.network.functions import hypothesis
from classifier.network.functions import computeGradient
from classifier.network.functions import reform

def train_with_bfgs(x,y,theta):


    print("training with bfgs")
    for i in range(np.shape(x)[0]):
        res = minimize(hypothesis.hypothesis, theta, args=(x[i],y[i]), method='SLSQP', jac=computeGradient.computeGradient,
                   options={'disp': True}) #outputs converged theta
        print("converged???")
