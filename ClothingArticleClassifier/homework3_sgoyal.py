import numpy as np
import matplotlib.pyplot as plt
import math

# Converts labels into one hot vectors
def getOneHotVectors(y):
    oneHotVectors = np.zeros((y.size,y.max()+1))
    oneHotVectors[np.arange(y.size),y] = 1
    return oneHotVectors

# Computes the Cross Entropy Loss
def fCE(y,yhat):
    return -np.sum(y*np.log(yhat))/y.shape[0]

# Computes the Percent Correct Accuracy
def fPC(y,yhat):
    labelY = np.argmax(y,axis=1)
    labelYhat = np.argmax(yhat,axis=1)
    return np.mean(labelY == labelYhat)
# Computes the gradient of the Cross Entropy Loss, will be used for softmax regression
def gradfCE(w, XT, y):
    n = len(y)
    z = XT.dot(w) # n X 10
    yhat = softMaxActivation(z)
    X = XT.T
    return X.dot(yhat - y)/n # Needs softmax

# Performs the softmax activation on the z values (pre-activation scores)
def softMaxActivation(z):
    A = np.exp(z)
    B = np.sum(A, axis=1).reshape(len(z),1)
    # print("z",z)
    # print("A ",A)
    # print("B ",B)
    return A/B
    # print(z.shape)
    # print(np.sum(np.exp(z), axis = 1) )
    # return np.exp(z)/(np.sum(np.exp(z), axis = 1).reshape(len(z),1))

# Given an array of faces (N x M , where N is number of examples and M is number of pixes),
# return a design matrix Xtilde ((M + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (X):
    ones = np.ones(X.shape[0]) # array for bias terms for each image
    reshapedX = X.T
    Xtilde = np.vstack((reshapedX,ones)) # Adding 1s for the bias term
    return Xtilde

# Randomly shuffles the data
def randomizeData(X,y):
    n = len(y)
    randomizedIndex = np.random.permutation(n)
    return X.T[randomizedIndex].T,y[randomizedIndex]

# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None):
    epochs = 50

    # Randomize order of the training data
    X,y = randomizeData(trainingImages,trainingLabels)

    # Initialize random weights with a bias = 1 for each category
    # w = np.random.normal(0, 0.01, (X.shape[0]-1,10))
    w = 0.01*np.random.randn(X.shape[0]-1,10)
    w = np.vstack((w,np.ones((1,10))))
    n = len(y)
    batches = math.ceil(n/batchSize)
    # print(batches)
    
    # print("w:",w)
    for i in range(epochs):
        # X,y = randomizeData(X,y)
        for j in range(batches):
            startIndex = j*batchSize
            endIndex = (j*batchSize)+100
            w = w - (epsilon*gradfCE(w,X.T[startIndex:endIndex],y[startIndex:endIndex]))

            if(j >= (batches - 20)):
                yhat = softMaxActivation(X.T[startIndex:endIndex].dot(w))
                print("Batch number:",j+1)
                print("Training Loss (fCE):",fCE(y[startIndex:endIndex],yhat))
                print()
            # break

    return w

if __name__ == "__main__":
    # Load data
    trainingImages = np.load("fashion_mnist_train_images.npy") # 60000 X 784 (28 X 28)
    trainingLabels = np.load("fashion_mnist_train_labels.npy")
    testingImages = np.load("fashion_mnist_test_images.npy") # 10000 X 784 (28 X 28)
    testingLabels = np.load("fashion_mnist_test_labels.npy")

    
    # Converting labels into one hot vectors
    yTraining = getOneHotVectors(trainingLabels) # 60000 X 10
    yTesting = getOneHotVectors(testingLabels)

    # Append a constant 1 term to each example to correspond to the bias terms as well as scale all the other values by 255
    Xtilde_tr = reshapeAndAppend1s(trainingImages/255) #  785 X 60000 
    Xtilde_te = reshapeAndAppend1s(testingImages/255) #   785 X 10000
    
    # print(list(Xtilde_tr.T[0]))
    # y = np.array([[0,1,0,0,0],[0,1,0,0,0]])
    # yhat = np.array([[0,1,0,0,0],[1,0,0,0,0]])
    # print(fCE(y,yhat))
    # print(fPC(y,yhat))
    # X,y = randomizeData(Xtilde_tr,yTraining)

    # Training the model
    W = softmaxRegression(Xtilde_tr, yTraining, Xtilde_te, yTesting, epsilon=0.1, batchSize=100)
    # print(W)

    yhatTraining = softMaxActivation(Xtilde_tr.T.dot(W))
    print("Training Accuracy (fPC):",fPC(yTraining,yhatTraining))

    yhatTesting = softMaxActivation(Xtilde_te.T.dot(W))
    print("Testing Accuracy (fPC):",fPC(yTesting,yhatTesting))
    print("Testing Loss (fCE):",fCE(yTesting,yhatTesting))

    # z = np.array([[1, 3, 5],
    #               [100,200,300]])
    # print(softMaxActivation(z))

    # Visualize the vectors
    
    for i in range(10):
        img = W.T[i][:-1].reshape(28,28)
        plt.imshow(img)
        plt.show()
    

