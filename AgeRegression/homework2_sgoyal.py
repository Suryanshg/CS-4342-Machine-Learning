import numpy as np

########################################################################################################################
# PROBLEM 2
########################################################################################################################
# Given a vector x of (scalar) inputs and associated vector y of the target labels, and given
# degree d of the polynomial, train a polynomial regression model and return the optimal weight vector.
def trainPolynomialRegressor (x, y, d):
    pass

########################################################################################################################
# PROBLEM 1
########################################################################################################################

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    ones = np.ones(faces.shape[0]) # array for bias terms for each image
    reshapedFaces = faces.reshape(faces.shape[0],faces.shape[1]*faces.shape[2]).T # reshaping to make each image a column vector
    Xtilde = np.vstack((reshapedFaces,ones)) # Adding 1s for the bias term
    return Xtilde

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):
    y_hat = (Xtilde.T).dot(w)
    squaredError = np.sum((y_hat - y)**2)
    meanSquaredError = squaredError/(2*len(y))
    return meanSquaredError

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    # Formula for gradient is (X((X.T)w - y))/n
    # Let A = (X.T)w - y, so gradientMse = X.dot(A)/n
    n = y.shape
    A = ((Xtilde.T).dot(w)) - y
    gradientMSE = (Xtilde.dot(A))/n
    return gradientMSE

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    # Using the form of equation A x = B, solving for x
    A = Xtilde.dot(Xtilde.T)
    B = Xtilde.dot(y)
    w = np.linalg.solve(A,B) # the solution x to the linear eq above which is x = Inv(A) . B
    return w

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    w = gradientDescent(Xtilde,y)
    return w

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    pass

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate = 0.003
    T = 5000  # Number of gradient descent iterations
    w = 0.01*np.random.randn(Xtilde.shape[0])
    for i in range(T):
        w = w - (EPSILON*gradfMSE(w, Xtilde,y,alpha))
    return w

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    # print(Xtilde_tr.shape)
    
    # Computing weights using different methods
    w1 = method1(Xtilde_tr, ytr) # Analytical Method
    w2 = method2(Xtilde_tr, ytr) # Gradient Descent Method
    w3 = method3(Xtilde_tr, ytr)

    print("Method 1: Analytical Method")
    trainingAccuracy1 = fMSE(w1,Xtilde_tr, ytr)
    testingAccuracy1 = fMSE(w1,Xtilde_te,yte)
    print("Training Accuracy:",trainingAccuracy1)
    print("Testing Accuracy:",testingAccuracy1)
    print()
    # print(w1.shape)

    # print(gradientDescent(Xtilde_tr,ytr))
    print("Method 2: Gradient Descent Method")
    trainingAccuracy2 = fMSE(w2,Xtilde_tr, ytr)
    testingAccuracy2 = fMSE(w2,Xtilde_te,yte)
    print("Training Accuracy:",trainingAccuracy2)
    print("Testing Accuracy:",testingAccuracy2)
    print()


    # Report fMSE cost using each of the three learned weight vectors
    # ...
