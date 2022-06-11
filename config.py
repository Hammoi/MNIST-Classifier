
#Training
network_size = (397,1) #neurons per layer, number of HIDDEN layers #TODO: cannot change layers, must fix
random_theta = True
iterations = 100
samples = 10000 #up to 60,000
available_samples = 60000
cv_samples = 100
available_cv_samples = 10000
lamb = 0.01 #for regularization of gradient
#TODO: the number of neurons per layer currently has to be greater than the number of input and output units; this can be fixed but im lazy

input = 784 #number of input units
output = 10 #number of output units

theta_dir = "data/network/theta.txt"
