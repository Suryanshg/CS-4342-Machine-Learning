import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt
import math

# Takes the Training Data (n X 2) and outputs the Transformed Training Data (n X 10)
def phiPoly3(x):
    r,a = x[:,0],x[:,1]
    n = x.shape[0]
    ones = np.ones(n)
    # r,a = x[0],x[1]
    return np.array([ones, math.sqrt(3)*r, math.sqrt(3)*a, math.sqrt(6)*r*a, math.sqrt(3)*r*r, math.sqrt(3)*a*a, math.sqrt(3)*r*r*a, math.sqrt(3)*r*a*a, r**3, a**3]).T

# Transforms the 
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

    
    # Show scatter-plot of the data
    '''
    idxsNeg = np.nonzero(y == -1)[0]
    idxsPos = np.nonzero(y == 1)[0]
    plt.scatter(X[idxsNeg, 0], X[idxsNeg, 1]) # Plotting negative examples
    plt.scatter(X[idxsPos, 0], X[idxsPos, 1]) # Plotting positive examples
    plt.title("Scatterplot of Training Data")
    plt.show()
    '''

    # print(X.shape)

    # (a) Train linear SVM using sklearn
    
    svmLinear = sklearn.svm.SVC(kernel='linear', C=0.01)
    svmLinear.fit(X, y)
    radons,asbestos =  np.meshgrid(np.arange(0,10,0.1),np.arange(54,186))
    # A, B = np.meshgrid(X,X)
    # print(A.T.reshape(100))
    # print(B.T.shape)
    # print(radons.reshape(132*11,1))
    # print(asbestos.reshape(132*11,1))
    Xtest = np.hstack((radons.reshape(radons.shape[0]*radons.shape[1],1),asbestos.reshape(asbestos.shape[0]*asbestos.shape[1],1)))
    yTest = svmLinear.predict(Xtest)
    showPredictions("Linear", Xtest, yTest)
    
    # (b) Poly-3 using explicit transformation phiPoly3
    '''
    print(X)
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
    '''

    Xtilde_tr = phiPoly3(X)

    # (c) Poly-3 using kernel matrix constructed by kernel function kerPoly3
    
    KTrain = []
    for x in X:
        row = []
        for xprime in X:
            row.append(kerPoly3(x,xprime))
        KTrain.append(np.array(row))
    KTrain = np.array(KTrain)
    
    # print(KTrain.shape)

    '''
    A = np.array([[1,2],[3,4],[5,6]])
    # AA, BB = np.meshgrid(A,A)
    # print(A)
    # print(">>>",AA)
    # print(">>>",BB)
    B1 = np.repeat(A,3,axis=0).reshape(3,3,2)
    print(B1)
    B2 = np.swapaxes(B1,0,1)
    print(B2)
    C = np.sum(B1*B2,axis=2)
    print(np.power(C,3))
    '''
    B1 = np.repeat(X,X.shape[0],axis=0).reshape(X.shape[0],X.shape[0],X.shape[1])
    B2 = np.swapaxes(B1,0,1)
    KTrain2 = np.power(1 + np.sum(B1*B2,axis=2),3)
    # print(KTrain = KTrain2)
    
    # TODO Kernelize Ktest
    KTest = []
    for x in Xtest:
        row = []
        for xprime in X:
            row.append(kerPoly3(x,xprime))
        KTest.append(np.array(row))
    KTest = np.array(KTest)
    print(KTest.shape)
    

    
    
    svmKernelTransform = sklearn.svm.SVC(kernel = 'precomputed')
    svmKernelTransform.fit(KTrain2,y)
    yTest = svmKernelTransform.predict(KTest)
    showPredictions("Poly Kernel", Xtest, yTest)
    
    

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

