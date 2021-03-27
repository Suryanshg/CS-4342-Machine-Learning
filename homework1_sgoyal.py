import numpy as np

def problem1 (A, B):
    return A + B

def problem2 (A, B, C): # Produces AB - C
    return (A.dot(B)) - C

def problem3 (A, B, C): # Produces A*B + C.T
    return (A*B) + C.T

def problem4 (x, S, y): # Produces (x.T)Sy
    return ((x.T).dot(S)).dot(y)

def problem5 (A): # Produces a zero matrix with same dimensions as A
    return np.zeros(A.shape)

def problem6 (A): # Produces a vector with same number of rows as A but contains all ones
    return np.ones((A.shape[0],))

def problem7 (A, alpha): # Produces A + (alpha)I
    return A + (alpha * np.eye(A.shape[0]))

def problem8 (A, i, j): # Produces j th column of the i th row
    return A[i,j]

def problem9 (A, i): # Produces sum of all the entries in the ith row in A
    return np.sum(A,axis=1)[i]

def problem10 (A, c, d): # Produces mean of all the entries in A between c and d (inclusive)
    S = A[np.nonzero((A<=d)&(A>=c))]
    return np.mean(S)

def problem11 (A, k): # Produces an (n x k) matrix comprising of right eigenvectors of Acorresponding to the k largest eigenvalues
    eValues, eVectors = np.linalg.eig(A)
    i = eValues.argsort()[::-1] # Sort in descending order
    eVectors = eVectors[:,i]
    return eVectors[:,:k]

def problem12 (A, x): # Produces the solution of the equations A y = x, which is same as y = Inv(A)x
    return np.linalg.solve(A,x)

def problem13 (A, x): # Produces x(Inv(A))
    # Given XY = ((Y.T (X.T))).T
    # Let X = x (row vector), Y = Inv(A), then X.T = x (column vector) Y.T = Inv(A.T)
    # So we just use the idea in problem 12
    result = np.linalg.solve(A.T,x) # Computes Inv(A.T)(x), so we just need to return its transpose
    return result.T

# Tester code
A = np.array([[1,2],
              [3,4]])
B = np.array([1,2])
C = np.array([[1,1,1],
              [4,5,6],
              [7,8,9]])
print(problem11(A,1))
