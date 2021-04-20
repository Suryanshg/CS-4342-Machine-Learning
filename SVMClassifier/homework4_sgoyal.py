from cvxopt import solvers, matrix
import numpy as np
import sklearn.svm

class SVM4342 ():
    def __init__ (self):
        pass

    # Expects each *row* to be an m-dimensional row vector. X should
    # contain n rows, where n is the number of examples.
    # y should correspondingly be an n-vector of labels (-1 or +1).
    def fit (self, X, y):

        Xtilde = np.hstack((X,np.ones((X.shape[0],1))))

        # np.arrays representing matrices or vectors for solving the QP problem
        G = -1*y.reshape(y.shape[0],1)*Xtilde # G = -1 * Y * Xtilde

        P = np.eye(Xtilde.shape[1]) # Identity Matrix

        q = np.zeros(Xtilde.shape[1]) # Zeros Matrix

        h = -1*np.ones(Xtilde.shape[0]) # Negative Ones

        # Solve -- if the variables above are defined correctly, you can call this as-is:
        sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))

        # Fetch the learned hyperplane and bias parameters out of sol['x']
        # To avoid any annoying errors due to broadcasting issues, I recommend
        # that you flatten() the w you retrieve from the solution vector so that
        # it becomes a 1-D np.array.
        w = np.array(sol['x']) # (m + 1) X 1
        self.w = w[:-1].reshape(1,w.shape[0]-1)  
        self.b = w[-1]  

    # Given a 2-D matrix of examples X, output a vector of predicted class labels
    def predict (self, x):
        yhat = x.dot(self.w.reshape(x.shape[1],1)) + self.b
        yhat[yhat < 0] = -1
        yhat[yhat > 0] = 1
        return yhat.reshape(len(yhat))

def test1 ():
    # Set up toy problem
    X = np.array([ [1,1], [2,1], [1,2], [2,3], [1,4], [2,4] ])
    y = np.array([-1,-1,-1,1,1,1])

    # Train your model
    svm4342 = SVM4342()
    svm4342.fit(X, y)
    print(svm4342.w, svm4342.b)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard-margin
    svm.fit(X, y)
    print(svm.coef_, svm.intercept_)

    acc = np.mean(svm4342.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

def test2 (seed):
    np.random.seed(seed)

    # Generate random data
    X = np.random.rand(20,3)
    # Generate random labels based on a random "ground-truth" hyperplane
    while True:
        w = np.random.rand(3)
        y = 2*(X.dot(w) > 0.5) - 1
        # Keep generating ground-truth hyperplanes until we find one
        # that results in 2 classes
        if len(np.unique(y)) > 1:
            break

    svm4342 = SVM4342()
    svm4342.fit(X, y)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard margin
    svm.fit(X, y)
    diff = np.linalg.norm(svm.coef_ - svm4342.w) + np.abs(svm.intercept_ - svm4342.b)
    print("Difference:",diff)

    acc = np.mean(svm4342.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

    if acc == 1 and diff < 1e-1:
        print("Passed test2")

if __name__ == "__main__": 

    # Tester code

    # X = np.array([ [1,1], [2,1], [1,2], [2,3], [1,4], [2,4] ])
    # y = np.array([-1,-1,-1,1,1,1])
    # Xtilde = np.hstack((X,np.ones((X.shape[0],1))))
    
    # print(-1*y.reshape(y.shape[0],1)*Xtilde)
    

    test1()
    for seed in range(5):
        test2(seed)
