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

def problem10 (A, c, d): 
    return ...

def problem11 (A, k):
    return ...

def problem12 (A, x):
    return ...

def problem13 (A, x):
    return ...

# Tester code
A = np.array([[1,2],
              [3,4]])
B = np.array([1,2])
print(problem9(A,0))
