import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt
import math

# Takes the Training Data (n X 2) and outputs the Transformed Training Data (n X 10)
def phiPoly3(x):
    r,a = x[:,0],x[:,1]
    n = x.shape[0]
    ones = np.ones(n)
    return np.array([ones, math.sqrt(3)*r, math.sqrt(3)*a, math.sqrt(6)*r*a, math.sqrt(3)*r*r, math.sqrt(3)*a*a, math.sqrt(3)*r*r*a, math.sqrt(3)*r*a*a, r**3, a**3]).T

# Transforms the given data sets into a Kernel Matrix using kernel function of degree 3
def kerPoly3 (x, xprime):
    B1 = np.repeat(xprime,x.shape[0],axis=0).reshape(xprime.shape[0],x.shape[0],xprime.shape[1])
    B2 = np.repeat(x,xprime.shape[0],axis=0).reshape(x.shape[0],xprime.shape[0],x.shape[1])
    B2 = np.swapaxes(B2,0,1)
    K = np.power(1 + np.sum(B1*B2,axis=2),3)
    return K

# Transforms the given data sets into a Kernel Matrix using kernel function
def kerRBF(x, xprime, gamma):
    B1 = np.repeat(xprime,x.shape[0],axis=0).reshape(xprime.shape[0],x.shape[0],xprime.shape[1])
    B2 = np.repeat(x,xprime.shape[0],axis=0).reshape(x.shape[0],xprime.shape[0],x.shape[1])
    B2 = np.swapaxes(B2,0,1)
    C1 = np.power(B1 - B2,2)
    D = np.sum(C1,axis=2)
    
    K = np.exp(-1*gamma*D)
    return K

# Visualizes the predictions of the SVMs on the test data
def showPredictions (title, Xtest, yTest):  # feel free to add other parameters if desired

    idxsNeg = np.nonzero(yTest == -1)[0]
    idxsPos = np.nonzero(yTest == 1)[0]

    plt.scatter(Xtest[idxsNeg, 0], Xtest[idxsNeg, 1]) # negative examples
    plt.scatter(Xtest[idxsPos, 0], Xtest[idxsPos, 1]) # positive examples

    plt.xlabel("Radon")
    plt.ylabel("Asbestos")
    plt.legend(["No lung disease", "Lung disease" ])
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
    showPredictions("a. Linear", Xtest, yTest)
    
    # (b) Poly-3 using explicit transformation phiPoly3
    
    Xtilde_tr = phiPoly3(X)
    svmExplicitTransform = sklearn.svm.SVC(kernel='linear', C=0.01)
    svmExplicitTransform.fit(Xtilde_tr,y)
    Xtilde_te = phiPoly3(Xtest)
    # print(Xtilde_te.shape)
    yTest = svmExplicitTransform.predict(Xtilde_te)
    showPredictions("b. Poly Kernel using Explicit Transformation", Xtest, yTest)


    # (c) Poly-3 using kernel matrix constructed by kernel function kerPoly3
    
    '''
    KTrain = []
    for x in X:
        row = []
        for xprime in X:
            row.append(kerPoly3(x,xprime))
        KTrain.append(np.array(row))
    KTrain = np.array(KTrain)
    '''

    '''
    KTest = []
    for x in Xtest:
        row = []
        for xprime in X:
            row.append(kerPoly31(x,xprime))
        KTest.append(np.array(row))
    KTest = np.array(KTest)
    print(KTest.shape)
    ''' 
    
    # Tester Code

    '''
    A = np.array([[1,2],[3,4],[5,6]])
    B = np.array([[1,2],[3,4]])
    B1 = np.repeat(A,B.shape[0],axis=0).reshape(A.shape[0],B.shape[0],2)

    B2 = np.repeat(B,A.shape[0],axis=0).reshape(B.shape[0],A.shape[0],2)
    # print(B2)

    B2 = np.swapaxes(B2,0,1)
    C = np.sum(B1*B2,axis=2)
    print(C)
    '''

    KTrain = kerPoly3(X,X)

    KTest = kerPoly3(X,Xtest)
    
    svmKernelTransform = sklearn.svm.SVC(kernel = 'precomputed')
    svmKernelTransform.fit(KTrain,y)
    yTest = svmKernelTransform.predict(KTest)
    showPredictions("c. Poly Kernel using Kernel function kerPoly3", Xtest, yTest)

    
    # (d) Poly-3 using sklearn's built-in polynomial kernel
    
    svmPolyKernel = sklearn.svm.SVC(kernel='poly', gamma = 1, coef0=1, degree=3)
    svmPolyKernel.fit(X,y)
    yTest = svmPolyKernel.predict(Xtest)
    showPredictions("d. Built-in Poly Kernel", Xtest, yTest)
    
    # (e) RBF using sklearn's built-in polynomial kernel
    
    svmRBF1 = sklearn.svm.SVC(kernel = 'rbf', gamma = 0.1)
    svmRBF1.fit(X,y)
    yTest = svmRBF1.predict(Xtest)
    showPredictions("e. RBF - gamma = 0.1",Xtest,yTest)

 
    preComputedSvmRBF1 = sklearn.svm.SVC(kernel = 'precomputed')
    KTrain = kerRBF(X,X,0.1)    
    KTest = kerRBF(X,Xtest,0.1)
    preComputedSvmRBF1.fit(KTrain,y)
    yTest = preComputedSvmRBF1.predict(KTest)
    showPredictions("precomputed Kernel RBF - gamma = 0.1",Xtest,yTest)

    
    svmRBF2 = sklearn.svm.SVC(kernel = 'rbf', gamma = 0.03)
    svmRBF2.fit(X,y)
    yTest = svmRBF2.predict(Xtest)
    showPredictions("e. RBF - gamma = 0.03",Xtest,yTest)

    preComputedSvmRBF2 = sklearn.svm.SVC(kernel = 'precomputed')
    KTrain = kerRBF(X,X,0.03)    
    KTest = kerRBF(X,Xtest,0.03)
    preComputedSvmRBF2.fit(KTrain,y)
    yTest = preComputedSvmRBF2.predict(KTest)
    showPredictions("precomputed Kernel RBF - gamma = 0.03",Xtest,yTest)
    
    
    
    

