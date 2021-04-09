import numpy as np
import matplotlib.pyplot as plt

# Computes the Cross Entropy Loss
def fCE(y,yhat):
    return -np.sum(y*np.log(yhat))/y.shape[0]

# Computes the Percent Correct Accuracy
def fPC(y,yhat):
    labelY = np.argmax(y,axis=1)
    labelYhat = np.argmax(yhat,axis=1)
    return np.mean(labelY == labelYhat)
# Computes the gradient of the Cross Entropy Loss, will be used for softmax regression
def gradfCE(w, X, y):
    n = len(y)
    z = X.T.dot(w) # 60000 X 10
    yhat = softMaxActivation(z)
    return X.dot(yhat - y)/n # Needs softmax

# Converts labels into one hot vectors
def getOneHotVectors(y):
    oneHotVectors = np.zeros((y.size,y.max()+1))
    oneHotVectors[np.arange(y.size),y] = 1
    return oneHotVectors

# Performs the softmax activation on the z values (pre-activation scores)
def softMaxActivation(z):
    return np.exp(z)/np.sum(np.exp(z), axis = 1) # Need more work

# Given an array of faces (N x M , where N is number of examples and M is number of pixes),
# return a design matrix Xtilde ((M + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (X):
    ones = np.ones(X.shape[0]) # array for bias terms for each image
    reshapedX = X.T
    Xtilde = np.vstack((reshapedX,ones)) # Adding 1s for the bias term
    return Xtilde

# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None):
    pass

if __name__ == "__main__":
    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy") # 60000 X 784 (28 X 28)
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy") # 10000 X 784 (28 X 28)
    testingLabels = np.load("fashion_mnist_test_labels.npy")

    
    # Converting labels into one hot vectors
    yTraining = getOneHotVectors(trainingLabels)
    yTesting = getOneHotVectors(testingLabels)

    # Append a constant 1 term to each example to correspond to the bias terms
    Xtilde_tr = reshapeAndAppend1s(trainingImages) #  785 X 60000 
    Xtilde_te = reshapeAndAppend1s(testingImages) #   785 X 10000

    
    # y = np.array([[0,1,0,0,0],[0,1,0,0,0]])
    # yhat = np.array([[0,1,0,0,0],[1,0,0,0,0]])
    # print(fCE(y,yhat))
    # print(fPC(y,yhat))

    # Training the model
    W = softmaxRegression(trainingImages, trainingLabels, testingImages, testingLabels, epsilon=0.1, batchSize=100)
    
    # Visualize the vectors
    # ...
