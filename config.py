
#Training
network_size = (397,1) #neurons per layer, number of HIDDEN layers #TODO: cannot change layers, must fix
random_theta = False
iterations = 100
split_data = True #split gradient computation up into multiple iterations (if memory error)
splices = 4 #must equal whole number when dividing samples
samples = 60000 #up to 60,000
available_samples = 60000
cv_samples = 10000
available_cv_samples = 10000
overwrite_theta_per_iteration = True #write theta to file after every iteration
lamb = 0.1 #for regularization of gradient
training = False #Train network?
test_cv = True #Test cross validation data?
test_train = True #Test with trained data?
start_gui = True #Start manual testing interface?


input = 784 #number of input units
output = 10 #number of output units

theta_dir = "data/network/theta.txt"
temp_theta_dir = "data/network/temp_theta.txt"
