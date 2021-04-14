import pandas
import numpy as np
import math

# Computes the Cross Entropy Loss
def fCE(y,yhat):
    return -np.sum(y*np.log(yhat))/y.shape[0]

# Computes the Percent Correct Accuracy
def fPC(y,yhat):
    labelY = np.argmax(y,axis=1)
    labelYhat = np.argmax(yhat,axis=1)
    return np.mean(labelY == labelYhat)

# Converts labels into one hot vectors
def getOneHotVectors(y):
    oneHotVectors = np.zeros((y.size,y.max()+1))
    oneHotVectors[np.arange(y.size),y] = 1
    return oneHotVectors

# Converts one hot vectors to labels
def getLabels(oneHotVectors):
    return np.argmax(oneHotVectors,axis=1)

# Computes the gradient of the Cross Entropy Loss, will be used for softmax regression
def gradfCE(w, XT, y):
    n = len(y)
    z = XT.dot(w) # n X 2
    yhat = softMaxActivation(z)
    X = XT.T
    return X.dot(yhat - y)/n

# Performs the softmax activation on the z values (pre-activation scores)
def softMaxActivation(z):
    A = np.exp(z)
    B = np.sum(A, axis=1).reshape(len(z),1)
    return A/B

# Randomly shuffles the data
def randomizeData(X,y):
    n = len(y)
    randomizedIndex = np.random.permutation(n)
    return X.T[randomizedIndex].T,y[randomizedIndex]

# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression (trainingData, trainingLabels, epsilon = None, batchSize = None): # batch size 9
    epochs = 30
    X,y = trainingData, trainingLabels
    
    # Initialize random weights with a bias = 1 for each category, there are two categories here
    # w = np.random.normal(0, 0.01, (X.shape[0]-1,10))
    w = 0.01*np.random.randn(X.shape[0]-1,2)
    w = np.vstack((w,np.ones((1,2))))
    n = len(y)
    batches = math.ceil(n/batchSize)
    # print(batches)
    
    # print("w:",w)  
    for i in range(epochs):
        X,y = randomizeData(X,y)
        for j in range(batches):
            startIndex = j*batchSize
            endIndex = (j*batchSize)+batchSize
            w = w - (epsilon*gradfCE(w,X.T[startIndex:endIndex],y[startIndex:endIndex]))
            
            '''
            if((i == epochs-1) and (j >= (batches - 20))):
                yhat = softMaxActivation(X.T[startIndex:endIndex].dot(w))
                print("Batch number:",j+1,"| Training Loss (fCE):",fCE(y[startIndex:endIndex],yhat))
                # print("Training Loss (fCE):",fCE(y[startIndex:endIndex],yhat))
                print()
            '''
            # break
            
    return w

if __name__ == "__main__":
    # Load training data
    d = pandas.read_csv("train.csv")
    y = d.Survived.to_numpy()
    sex = d.Sex.map({"male":0, "female":1}).to_numpy()
    Pclass = d.Pclass.to_numpy()
    SibSp = d.SibSp.to_numpy() # SibSp is ordinal

    # print("y:",y)
    # print("sex:",getOneHotVectors(sex))
    # print("Pclass:",Pclass)
    # print("SibSp:",SibSp)

    sex = getOneHotVectors(sex) # Sex is categorical
    Pclass = getOneHotVectors(Pclass)[:,1:] # Pclass is categorical

    yTraining = getOneHotVectors(y)
    nTraining = len(d)
    Xtilde_tr = np.hstack((sex,Pclass,SibSp.reshape(nTraining,1),np.ones((nTraining,1)))).T # 7 X 891
    
    # print(Xtilde_tr.shape)
    # print(Xtilde_tr.T[0])

    # Train model using part of homework 3.
    # Dimensions of Weights should be 7 X 2
    
    W = softmaxRegression(Xtilde_tr,yTraining,epsilon = 0.1,batchSize = 9)
    
    print("Weights:",W)

    yhatTraining = softMaxActivation(Xtilde_tr.T.dot(W))
    print("Training Accuracy (fPC):",fPC(yTraining,yhatTraining))

    # Load testing data
    d = pandas.read_csv("test.csv")
    # y = d.Survived.to_numpy()
    sex = getOneHotVectors(d.Sex.map({"male":0, "female":1}).to_numpy())
    Pclass = getOneHotVectors(d.Pclass.to_numpy())[:,1:]
    SibSp = d.SibSp.to_numpy()
    PassengerId = d.PassengerId.to_numpy()

    # yTesting = getOneHotVectors(y)
    nTesting = len(d)
    Xtilde_te = np.hstack((sex,Pclass,SibSp.reshape(nTesting,1),np.ones((nTesting,1)))).T # 7 X 418

    # Compute predictions on test set
    yhatTesting = softMaxActivation(Xtilde_te.T.dot(W))
    yhatLabels = getLabels(yhatTesting)
    # print(yhatLabels)
    # print(PassengerId)

    # Write CSV file of the format:
    # PassengerId, Survived
    df = pandas.DataFrame({'PassengerId':PassengerId, 'Survived':yhatLabels})
    # df.to_csv('predictions.csv',index = False)

    # Current score: 0.77272
    