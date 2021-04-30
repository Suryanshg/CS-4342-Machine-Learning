import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

# Relu Activation function
def relu(z):
    pass

# Softmax Activation function
def softmax(z):
    pass

# Relu prime on the input
def reluprime(z):
    pass

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    endW1 = NUM_HIDDEN * NUM_INPUT
    endb1 = endW1 + NUM_HIDDEN
    endW2 = endb1 + (NUM_OUTPUT * NUM_HIDDEN)
    W1 = w[:endW1].reshape(NUM_HIDDEN, NUM_INPUT)
    b1 = w[endW1:endb1].reshape(NUM_HIDDEN)
    W2 = w[endb1:endW2].reshape(NUM_OUTPUT, NUM_HIDDEN)
    b2 = w[endW2:].reshape(NUM_OUTPUT)
    return W1, b1, W2, b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    w = np.concatenate([W1.flatten(),b1.flatten(),W2.flatten(), b2.flatten()])
    return w

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("fashion_mnist_{}_images.npy".format(which))
    labels = np.load("fashion_mnist_{}_labels.npy".format(which))
    return images, labels

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    return cost

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    return grad

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train (trainX, trainY, testX, testY, w):
    pass

# Optimize the hyperparameters
def findBestHyperparameters():
    pass

# Randomly shuffles the data and returns 20% of that as validation data
def getValidationData(X,y):
    n = len(y)
    randomizedIndex = np.random.permutation(n)
    valX, valY = X[randomizedIndex][:n//5],y[randomizedIndex][:n//5]
    return valX, valY

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train") # (60000,784) , (60000,)
        testX, testY = loadData("test") # (10000,784), (10000,)

    # Generating Validation Data
    valX, valY = getValidationData(trainX, trainY)

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5 # (40,784)
    b1 = 0.01 * np.ones(NUM_HIDDEN) # (40,)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5 # (10,40)
    b2 = 0.01 * np.ones(NUM_OUTPUT) #(10,)

    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)


    '''
    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    w))
    '''

    # Hyperparameter Tuning


    # Train the network using SGD.
    train(trainX, trainY, testX, testY, w)
