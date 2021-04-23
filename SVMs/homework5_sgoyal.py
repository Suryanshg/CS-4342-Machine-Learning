import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt

def phiPoly3 (x):
    pass

def kerPoly3 (x, xprime):
    pass

def showPredictions (title, svm, X):  # feel free to add other parameters if desired
    radons,asbestos =  np.meshgrid(np.arange(0,10,0.1),np.arange(54,186))
    # print(radons.reshape(132*11,1))
    # print(asbestos.reshape(132*11,1))
    Xtest = np.hstack((radons.reshape(radons.shape[0]*radons.shape[1],1),asbestos.reshape(asbestos.shape[0]*asbestos.shape[1],1)))
    yTest = svm.predict(Xtest)

    idxsNeg = np.nonzero(yTest == -1)[0]
    idxsPos = np.nonzero(yTest == 1)[0]

    plt.scatter(Xtest[idxsNeg, 0], Xtest[idxsNeg, 1]) # negative examples
    plt.scatter(Xtest[idxsPos, 0], Xtest[idxsPos, 1]) # positive examples

    plt.xlabel("Radon")
    plt.ylabel("Asbestos")
    plt.legend([ "Lung disease", "No lung disease" ])
    plt.title(title)
    plt.show()
    

if __name__ == "__main__":
    # Load training data
    d = np.load("lung_toy.npy")
    X = d[:,0:2]  # features
    y = d[:,2]  # labels

    
    # Get max and min for each axis
    # print(np.min(X[:,1]))

    # Show scatter-plot of the data
    idxsNeg = np.nonzero(y == -1)[0]
    idxsPos = np.nonzero(y == 1)[0]
    plt.scatter(X[idxsNeg, 0], X[idxsNeg, 1]) # Plotting negative examples
    plt.scatter(X[idxsPos, 0], X[idxsPos, 1]) # Plotting positive examples
    plt.title("Scatterplot of Training Data")
    plt.show()

    # (a) Train linear SVM using sklearn
    svmLinear = sklearn.svm.SVC(kernel='linear', C=0.01)
    svmLinear.fit(X, y)
    showPredictions("Linear", svmLinear, X)

    # (b) Poly-3 using explicit transformation phiPoly3
    
    # (c) Poly-3 using kernel matrix constructed by kernel function kerPoly3
    
    # (d) Poly-3 using sklearn's built-in polynomial kernel

    # (e) RBF using sklearn's built-in polynomial kernel
