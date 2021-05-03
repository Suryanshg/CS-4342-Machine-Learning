import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import math

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

# Converts labels into one hot vectors
def getOneHotVectors(y):
    oneHotVectors = np.zeros((y.size,y.max()+1))
    oneHotVectors[np.arange(y.size),y] = 1
    return oneHotVectors

# Relu Activation function, z = (40,n)
def relu(z):
    z[z<=0] = 0
    return z

# Softmax Activation function, z = (10,n)
def softmax(z):
    zT = z.T # (n,10)
    # A = np.exp(z)
    # B = np.sum(A, axis = 0)
    return (np.exp(zT)/(np.sum(np.exp(zT), axis = 1).reshape(len(zT),1))).T


# Relu prime activation on the input, z = (40,n)
def reluprime(z):
    z[z<=0] = 0
    z[z>0] = 1
    return z

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
# TODO: add param hidden units
def unpack (w, hiddenUnits):
    endW1 = hiddenUnits * NUM_INPUT
    endb1 = endW1 + hiddenUnits
    endW2 = endb1 + (NUM_OUTPUT * hiddenUnits)
    W1 = w[:endW1].reshape(hiddenUnits, NUM_INPUT)
    b1 = w[endW1:endb1].reshape(hiddenUnits)
    W2 = w[endb1:endW2].reshape(NUM_OUTPUT, hiddenUnits)
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

# Computes the Percent Correct Accuracy
def fPC(y,yhat):
    labelY = np.argmax(y,axis=1)
    labelYhat = np.argmax(yhat,axis=1)
    return np.mean(labelY == labelYhat)

# Calculates yhat for a given data and weights and biases associated
# TODO: add param hidden units, recheck
def getYHat(X,w,hiddenUnits):
    # Shape of X is (n,784)
    # TODO: add arg hidden units
    W1, b1, W2, b2 = unpack(w, hiddenUnits)

    b1 = b1.reshape(hiddenUnits,1)
    b2 = b2.reshape(NUM_OUTPUT,1)

    # Calculating according to the equations
    z1 = W1.dot(X) + b1 # (40,n)
    h = relu(z1) # (40,n)
    z2 = W2.dot(h) + b2
    yhatT = softmax(z2) # (10,n)
    
    return yhatT.T #(n,10)

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE (X, Y, w, hiddenUnits):
    yhat = getYHat(X,w,hiddenUnits) # (n,10)
    cost = -np.sum(Y * np.log(yhat))/len(Y)
    return cost

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
# TODO: add param hidden units
def gradCE (X, Y, w, hiddenUnits):

    # TODO: add arg hidden units
    W1, b1, W2, b2 = unpack(w,hiddenUnits)

    # X here is (784,n)
    b1 = b1.reshape(hiddenUnits,1) # (40,n)
    b2 = b2.reshape(NUM_OUTPUT,1) # (10,n)

    # Performing foward propagation according to the equations
    z1 = W1.dot(X) + b1 # (40,n)
    h = relu(z1) # (40,n)
    z2 = W2.dot(h) + b2
    yhatT = softmax(z2) # (10,n)

    # Backward Propagation
    # Calculating gradients according to the equations
    gradW2 =  (yhatT - Y.T).dot(h.T)/Y.shape[0] # (10,40)
    gradb2 = np.average((yhatT - Y.T), axis = 1).reshape(NUM_OUTPUT,1) # (10,1)

    gT = ((yhatT.T - Y).dot(W2))*reluprime(z1.T) # (n,40)
    g = gT.T # (40,n)

    gradW1 = g.dot(X.T)/Y.shape[0] # (40,784)
    gradb1 =  np.average(g, axis = 1).reshape(hiddenUnits,1) # (40,1)

    grad = pack(gradW1,gradb1,gradW2,gradb2)
    return grad

# Randomly shuffles the data
def randomizeData(X,y):
    n = len(y)
    randomizedIndex = np.random.permutation(n)
    return X.T[randomizedIndex].T,y[randomizedIndex]

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
# TODO: replace w param with hidden units
def train(trainX, trainY, hiddenUnits, epsilon, miniBatchSize, epochs, alpha):
    # trainX is (784,60000)

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(hiddenUnits, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5 # (40,784)
    b1 = 0.01 * np.ones(hiddenUnits) # (40,)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, hiddenUnits))/hiddenUnits**0.5) - 1./hiddenUnits**0.5 # (10,40)
    b2 = 0.01 * np.ones(NUM_OUTPUT) # (10,)
    
    w = pack(W1,b1,W2,b2)

    N = len(trainY)
    batches = math.ceil(N/miniBatchSize)
    
    X,y = randomizeData(trainX,trainY)
    for epoch in range(epochs):
        for j in range(batches):
            startIndex = j * miniBatchSize
            endIndex = (j * miniBatchSize) + miniBatchSize

            miniX = X.T[startIndex:endIndex].T
            miniY = y[startIndex:endIndex]

            # Extracting Weights and biases
            # TODO: add arg hidden units
            W1,b1,W2,b2 = unpack(w, hiddenUnits)
            b1 = b1.reshape(hiddenUnits,1)
            b2 = b2.reshape(NUM_OUTPUT,1)

            grad = gradCE(miniX,miniY,w,hiddenUnits)

            # TODO: add arg hidden units
            gradW1, gradb1, gradW2, gradb2 = unpack(grad, hiddenUnits)

            gradb1 = gradb1.reshape(hiddenUnits,1) # (40,1)
            gradb2 = gradb2.reshape(NUM_OUTPUT,1) # (10,1)
            
           
            # Updating the weights and biases
            W1 = W1 - epsilon * (gradW1 - ((alpha * W1)/len(miniY)))
            b1 = b1 - epsilon * (gradb1 - ((alpha * b1)/len(miniY)))
            W2 = W2 - epsilon * (gradW2 - ((alpha * W2)/len(miniY)))
            b2 = b2 - epsilon * (gradb2 - ((alpha * b2)/len(miniY)))

            w = pack(W1,b1,W2,b2)

    return w

# Optimize the hyperparameters
def findBestHyperparameters(trainX, trainY, valX, valY):
    bestNumUnitsInHiddenLayer = 0 
    bestEpsilon = 0
    bestMiniBatchSize = 0
    bestNumEpochs = 0
    # bestAlpha = 0 # Keep 0.01
    bestAcc = 0
    
    unitsInHiddenLayer = [30,40,50]
    epochs = [20, 30, 40]
    epsilons = [0.01, 0.05]
    miniBatchSizes = [32, 64, 128, 256]
    alpha = 0.01

    for hiddenUnits in unitsInHiddenLayer:
        for epsilon in epsilons:
            for miniBatchSize in miniBatchSizes:
                for epoch in epochs:

                    # TODO: replace arg w with hidden units
                    trainedW = train(trainX, trainY, hiddenUnits, epsilon, miniBatchSize, epoch, alpha)
                    valYHat = getYHat(valX, trainedW, hiddenUnits)

                    curAcc = fPC(valY, valYHat)
                    print("hiddenUnits: {}, epsilon: {}, batchSize: {}, epochs: {}, alpha: {}, accuracy: {}".format(hiddenUnits, epsilon, miniBatchSize, epoch, alpha, curAcc))

                    if curAcc > bestAcc:
                        bestAcc = curAcc
                        bestNumUnitsInHiddenLayer = hiddenUnits
                        bestEpsilon = epsilon
                        bestMiniBatchSize = miniBatchSize
                        bestNumEpochs = epoch
                        

    return  bestNumUnitsInHiddenLayer,bestEpsilon, bestMiniBatchSize, bestNumEpochs, alpha

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
    
    # Scaling and Transposing Input Data (784,n)
    trainX = trainX.T/255
    testX = testX.T/255
    valX = valX.T/255

    trainY = getOneHotVectors(trainY)
    testY = getOneHotVectors(testY)
    valY = getOneHotVectors(valY)
    
    # Initialize weights randomly
    # W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5 # (40,784)
    # b1 = 0.01 * np.ones(NUM_HIDDEN) # (40,)
    # W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5 # (10,40)
    # b2 = 0.01 * np.ones(NUM_OUTPUT) # (10,)

    # W1 = 2*(np.random.random(size=(30, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5 # (30,784)
    # b1 = 0.01 * np.ones(30) # (30,)
    # W2 = 2*(np.random.random(size=(NUM_OUTPUT, 30))/30**0.5) - 1./30**0.5 # (10,30)
    # b2 = 0.01 * np.ones(NUM_OUTPUT) # (10,)
    

    
    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    # w = pack(W1, b1, W2, b2)
    
    '''
    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    lambda w_: gradCE(np.atleast_2d(trainX[:,idxs]), np.atleast_2d(trainY[idxs,:]), w_), \
                                    w))
    '''
    
    
    # Hyperparameter Tuning
    print()
    bestNumUnitsInHiddenLayer, bestEpsilon, bestMiniBatchSize, bestNumEpochs, alpha = findBestHyperparameters(trainX, trainY, valX, valY)
    print("bestNumUnitsInHiddenLayer",bestNumUnitsInHiddenLayer)
    print("bestEpsilon", bestEpsilon)
    print("bestMiniBatchSize", bestMiniBatchSize)
    print("bestNumEpochs", bestNumEpochs)
    print("alpha", alpha)
    

    # trainedW = train(trainX,trainY, w, bestEpsilon, bestMiniBatchSize, bestNumEpochs, alpha)
    trainedW = train(trainX,trainY, bestNumUnitsInHiddenLayer, bestEpsilon, bestMiniBatchSize, bestNumEpochs, alpha)

    testYHat = getYHat(testX, trainedW, bestNumUnitsInHiddenLayer )
    print("Test Accuracy:",fPC(testY,testYHat)*100)
    # Train the network using SGD.
    # print(train(trainX, trainY, w, 0.01, 128, 10 ,0.01).shape)
    
    