import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt
import math

def phiPoly3(x):
    r,a = x[0],x[1]
    # return np.array([1, math.sqrt(3)*r, math.sqrt(3)*a, math.sqrt(6)*r*a, math.sqrt(3)*r*r, math.sqrt(3)*a*a, math.sqrt(3)*r*r*a, math.sqrt(3)*r*a*a, r**3, a**3])
    return [1, math.sqrt(3)*r, math.sqrt(3)*a, math.sqrt(6)*r*a, math.sqrt(3)*r*r, math.sqrt(3)*a*a, math.sqrt(3)*r*r*a, math.sqrt(3)*r*a*a, r**3, a**3]
    # return np.array([x[0],x[1],x[0]*x[1]])

def kerPoly3 (x, xprime):
    return math.pow(1+x.T.dot(xprime),3)

def showPredictions (title, Xtest, yTest):  # feel free to add other parameters if desired
    '''
    radons,asbestos =  np.meshgrid(np.arange(0,10,0.1),np.arange(54,186))
    # print(radons.reshape(132*11,1))
    # print(asbestos.reshape(132*11,1))
    Xtest = np.hstack((radons.reshape(radons.shape[0]*radons.shape[1],1),asbestos.reshape(asbestos.shape[0]*asbestos.shape[1],1)))
    yTest = svm.predict(Xtest)
    '''

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

    '''
    # Show scatter-plot of the data
    idxsNeg = np.nonzero(y == -1)[0]
    idxsPos = np.nonzero(y == 1)[0]
    plt.scatter(X[idxsNeg, 0], X[idxsNeg, 1]) # Plotting negative examples
    plt.scatter(X[idxsPos, 0], X[idxsPos, 1]) # Plotting positive examples
    plt.title("Scatterplot of Training Data")
    plt.show()
    '''

    # (a) Train linear SVM using sklearn
    svmLinear = sklearn.svm.SVC(kernel='linear', C=0.01)
    svmLinear.fit(X, y)
    radons,asbestos =  np.meshgrid(np.arange(0,10,0.1),np.arange(54,186))
    # print(radons.reshape(132*11,1))
    # print(asbestos.reshape(132*11,1))
    Xtest = np.hstack((radons.reshape(radons.shape[0]*radons.shape[1],1),asbestos.reshape(asbestos.shape[0]*asbestos.shape[1],1)))
    yTest = svmLinear.predict(Xtest)
    # showPredictions("Linear", Xtest, yTest)
    
    # (b) Poly-3 using explicit transformation phiPoly3

    Xtilde_tr = []
    for x in X:
        Xtilde_tr.append(phiPoly3(x))
    Xtilde_tr = np.array(Xtilde_tr)
    # Xtilde = phiPoly3(X)
    # print(Xtilde_tr.shape)

    svmExplicitTransform = sklearn.svm.SVC(kernel='linear', C=0.01)
    svmExplicitTransform.fit(Xtilde_tr,y)
    # print("Done training")
    Xtilde_te = []
    for x in Xtest:
        Xtilde_te.append(phiPoly3(x))
    Xtilde_te = np.array(Xtilde_te)
    # print(Xtilde_te.shape)
    yTest = svmExplicitTransform.predict(Xtilde_te)
    # showPredictions("Explicit Transformation", Xtest, yTest)
    # (c) Poly-3 using kernel matrix constructed by kernel function kerPoly3
    '''
    KTrain = []
    for x in X:
        row = []
        for xprime in X:
            row.append(kerPoly3(x,xprime))
        KTrain.append(np.array(row))
    KTrain = np.array(KTrain)
    KTest = []
    for x in Xtest:
        row = []
        for xprime in X:
            row.append(kerPoly3(x,xprime))
        KTest.append(np.array(row))
    KTest = np.array(KTest)
    
    svmKernelTransform = sklearn.svm.SVC(kernel = 'precomputed')
    svmKernelTransform.fit(KTrain,y)
    yTest = svmKernelTransform.predict(KTest)
    showPredictions("Poly Kernel", Xtest, yTest)
    '''

    # (d) Poly-3 using sklearn's built-in polynomial kernel
    '''
    svmPolyKernel = sklearn.svm.SVC(kernel='poly', gamma = 1, coef0=1, degree=3)
    svmPolyKernel.fit(X,y)
    yTest = svmPolyKernel.predict(Xtest)
    showPredictions("Built-in Poly Kernel", Xtest, yTest)
    '''
    # (e) RBF using sklearn's built-in polynomial kernel
    svmRBF1 = sklearn.svm.SVC(kernel = 'rbf', gamma = 0.1)
    svmRBF1.fit(X,y)
    yTest = svmRBF1.predict(Xtest)
    showPredictions("RBF- gamma = 0.1",Xtest,yTest)

    svmRBF2 = sklearn.svm.SVC(kernel = 'rbf', gamma = 0.03)
    svmRBF2.fit(X,y)
    yTest = svmRBF2.predict(Xtest)
    showPredictions("RBF- gamma = 0.03",Xtest,yTest)

