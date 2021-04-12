import pandas
import numpy as np

# Converts labels into one hot vectors
def getOneHotVectors(y):
    oneHotVectors = np.zeros((y.size,y.max()+1))
    oneHotVectors[np.arange(y.size),y] = 1
    return oneHotVectors

# Computes the gradient of the Cross Entropy Loss, will be used for softmax regression
def gradfCE(w, XT, y):
    n = len(y)
    z = XT.dot(w) # n X 10
    yhat = softMaxActivation(z)
    X = XT.T
    return X.dot(yhat - y)/n

# Performs the softmax activation on the z values (pre-activation scores)
def softMaxActivation(z):
    A = np.exp(z)
    B = np.sum(A, axis=1).reshape(len(z),1)
    # print("z",z)
    # print("A ",A)
    # print("B ",B)
    return A/B

# Randomly shuffles the data
def randomizeData(X,y):
    n = len(y)
    randomizedIndex = np.random.permutation(n)
    return X.T[randomizedIndex].T,y[randomizedIndex]

# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression (trainingImages, trainingLabels, testingImages, testingLabels, epsilon = None, batchSize = None):
    epochs = 10

    # Randomize order of the training data
    # X,y = randomizeData(trainingImages,trainingLabels)
    X,y = trainingImages, trainingLabels

    # Initialize random weights with a bias = 1 for each category
    # w = np.random.normal(0, 0.01, (X.shape[0]-1,10))
    w = 0.01*np.random.randn(X.shape[0]-1,10)
    w = np.vstack((w,np.ones((1,10))))
    n = len(y)
    batches = math.ceil(n/batchSize)
    # print(batches)
    
    # print("w:",w)
    for i in range(epochs):
        X,y = randomizeData(X,y)
        for j in range(batches):
            startIndex = j*batchSize
            endIndex = (j*batchSize)+100
            w = w - (epsilon*gradfCE(w,X.T[startIndex:endIndex],y[startIndex:endIndex]))
            
            if((i == epochs-1) and (j >= (batches - 20))):
                yhat = softMaxActivation(X.T[startIndex:endIndex].dot(w))
                print("Batch number:",j+1,"| Training Loss (fCE):",fCE(y[startIndex:endIndex],yhat))
                # print("Training Loss (fCE):",fCE(y[startIndex:endIndex],yhat))
                print()
            # break
            

    return w

if __name__ == "__main__":
    # Load training data
    d = pandas.read_csv("train.csv")
    y = d.Survived.to_numpy()
    sex = d.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass = d.Pclass.to_numpy()
    SibSp = d.SibSp.to_numpy()

    # print("y:",y)
    # print("sex:",sex)
    # print("Pclass:",Pclass)
    # print("SibSp:",SibSp)

    yTraining = getOneHotVectors(y)
    nTraining = len(d)
    Xtilde_tr = np.hstack((sex.reshape(nTraining,1),Pclass.reshape(nTraining,1),SibSp.reshape(nTraining,1),np.ones((nTraining,1)))).T # 4 X 891

    # Train model using part of homework 3.
    # Dimensions of Weights should be 4 X 2

    # Load testing data
    # ...

    # Compute predictions on test set
    # ...

    # Write CSV file of the format:
    # PassengerId, Survived
    # ..., ...
