import numpy as np
import config
from classifier.network.functions import hypothesis
from classifier.network.functions import reform

np.set_printoptions(edgeitems=30, linewidth=1000,
    formatter=dict(float=lambda x: "%.3g" % x))

def send(array): #28*28, flattened
    data = np.reshape(array, (28,28))
    print("sending")
    #print(data)
    thetas = reform.reform_theta(np.genfromtxt(config.theta_dir))


    hypo = hypothesis.hypothesis(thetas, array)
    print("network guess: {}".format(np.argmax(hypo)))
    print("with {}% certainty".format(np.max(hypo)/np.sum(hypo)*100))
    print("hypothesis: {}".format(hypo))
    return np.argmax(hypo)
