# MNIST-Classifier

## Goal
To create a neural network that is able to accuractely compute the value of any single-digit handwritten number. This will be done through training a network using the MNIST dataset of 60,000 training samples. The network will be tested for cross-validated accuracy through 10,000 testing samples from MNIST.

## The Network
The network is a three layer neural network with one input layer, one hidden layer, and one output layer. It takes any 28x28 image as input (hence 784 input values), and outputs its guess from the range [0,9] (hence 10 output values). The hidden layer has 397 neurons (this can be changed in config.py), with each using the sigmoid function as its activation function. The final hypothesis is computed through taking the argmax of its output layer's values.

## Training
The network is trained using gradient descent. Its gradients are manually computed with a vectorized self-written function and its training process is completely self-written. The gradients are regularized and then subtracted from the theta values. For the final training process, this was repeated for around two hundred iterations using 60,000 training data and trained on a Google Cloud compute engine.

## End Result
After its final training, the network ouputs a 94.921% accuracy with its training data and a 94.300% accuracy with its cross validation data (10,000 samples). This accuracy can definitely be improved upon by modifiying its hidden layer components in ways such as changing its number of hidden neurons, activation function, training method, and much more.

## Manual Testing
The program also includes a GUI for manually testing the accuracy of the network. The user is provided with a 28x28 "canvas" created through Tkinter that they are able to draw on. Upon drawing a single-digit value, the user can send this canvas to the network, which returns its hypothesis to the user through the GUI.
