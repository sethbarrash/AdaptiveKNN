import numpy  as np
import pandas as pd
from scipy import optimize as opt
from scipy import stats
from scipy.sparse import csgraph
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

## Functions for laplacians

def computeLaplacian(y):
    y = y.squeeze()
    A = np.logical_not(y - y[:, np.newaxis])    # Compute the adjacency for labels y
    A = np.array(A, dtype = int)                # Change adjacency to int
    L = csgraph.laplacian(A)                    # Compute the laplacian of the adjacency
    return L



def computeLaplacianLocal(X, y, k):
    """
    In this Laplacian, the entry aij is 1 if yi and yj are the same label and
    xi and xj are neighbors.  Otherwise, it is zero.

    Could potentially be sped up using sparse matrices.
    """

    y = y.squeeze()
    NN = NearestNeighbors()
    NN.fit(X)
    ## When calculating the nearest neighbor graph, we use the k input by the
    ## user plus one. That is because a point will always be its own nearest
    ## neighbor. This will not affect the regularizer because wii is
    ## constrained to be zero anyways.
    M = NN.kneighbors_graph(X, k + 1)
    M = M.toarray()
    A = np.logical_not(y - y[:, np.newaxis])    # Compute the adjacency for labels y
    A = np.array(A, dtype = int)                # Change adjacency to int

    A = A * M                                   # AND A with M
    L = csgraph.laplacian(A)                    # Compute the laplacian of the adjacency
    return L



def laplacianZhang(X, k, sigma):
    D = X.shape[1]
    Sfull = np.empty((D, D))
    S = np.zeros((D, D))

    # Calculate the heat kernels between all feature vectors
    for i in range(D):
        Sfull[i, i] = 1.
        for j in range(i + 1, D):
            xi = X[:, i]
            xj = X[:, j]
            kij = np.exp( -1. * np.sum((xi - xj) ** 2) / (2 * sigma) )
            Sfull[i, j] = kij
            Sfull[j, i] = kij

    # Get rid of all but the k largest elements in each row
    for i in range(D):
        topk = np.sort(Sfull[i, :])[-k:]
        idxk = np.argsort(Sfull[i, :])[-k:]
        S[i, idxk] = topk

    S = 0.5 * (S + S.T)

    D = np.diag(np.sum(S, axis = 0))
    
    return D - S






## Functions for ADMM


def updateZ(W, U, rho, c):
    condMat = W + U / c

    # Find which soft thresholding action is applicable to each index
    condp = condMat >  2 * rho / c
    #cond0 = np.logical_and(condMat <= rho / c, condMat >= -rho / c)
    #condn = condMat <  - 2 * rho / c

    # Calculate the soft-thresholded values
    Z = np.zeros_like(W)
    Z[condp] = condMat[condp] - 2 * rho / c
    #Z[cond0] = 0
    #Z[condn] = condMat[condn] + 2 * rho / c

    return Z




def admm(Kxx, Ainv, rho1, c, tol, thresh):
    N = Kxx.shape[0]
    W = np.zeros((N, N))                # Decision variable
    Z = np.zeros((N, N))                # Auxiliary variable
    U = np.zeros((N, N))                # Dual variable

    Wold      = np.zeros((N, N))        # Decision variable from previous iteration
    converged = False
    iters   = 0

    while not converged and iters < 1000:
        ## Update W
        B  = -2 * Kxx + U - c * Z
        W  = -0.5 * Ainv @ B
        W[W < 0] = 0
        np.fill_diagonal(W, 0)

        ## Update Z
        Z  = updateZ(W, U, rho1, c)

        ## Update U
        WZ = W - Z
        U  = U + c * WZ
        U[U < 0] = 0

        ## Approximately check optimality
        convCondition0 = np.max( np.abs(WZ) ) < tol
        convCondition1 = np.max( np.abs(W - Wold) ) < tol
        converged = convCondition0 and convCondition1

        Wold = W
        iters += 1

    Z[Z < thresh] = 0
    diverged = np.max(Z) > 1e3

    return Z, converged, diverged, iters







## Functions for kernels

def gramMatrix(X, polyweights, polyoffsets, rbfweights, rbfwidths):
    N = X.shape[0]
    
    # Initialize Gram matrix
    Kxx = np.zeros((N, N))

    # Add polynomial kernels to Gram matrix
    
    # Check if there are any polynomial kernels to add
    if polyweights:
        XX = X @ X.T
        order = 0

        for weight, offset in zip(polyweights, polyoffsets):
            order += 1
            if not weight: continue
            Knew = XX + offset
            Knew = Knew ** order
            Knew = Knew * weight
            Kxx += Knew
    

    # Add radial basis function kernels to Gram matrix

    # Check if there are any rbf kernels to add
    if rbfweights:
        # Map the data to a matrix whose entry (i, j) is ||xi - xj||^2
        XX = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                xi = X[i, :]
                xj = X[j, :]
                XX[i, j] = XX[j, i] = np.sum( (xi - xj) ** 2)

        for weight, width in zip(rbfweights, rbfwidths):
            if not weight: continue
            Knew = weight * np.exp(-XX / width)
            Kxx += Knew


    return Kxx




def predictKer(Ktest, Dxx, ytrain):
    yhat = np.zeros_like(Ktest)

    for i in range(len(yhat)):
        ki      = Ktest[i]
        topK    = np.argsort(Dxx[i])[:ki] # Take indices of the K nearest neighbors to the ith testing example
        classes = ytrain[topK]
        yhat[i] = stats.mode(classes)[0][0]

    return yhat




def distancesKer(X_train, X_valid, polyweights, polyoffsets, rbfweights, rbfwidths):
    m = X_valid.shape[0]
    n = X_train.shape[0]
    Dxx = np.zeros((m, n))

    XtXt = np.diag(X_train @ X_train.T)
    XvXv = np.diag(X_valid @ X_valid.T)
    XvXt = X_valid @ X_train.T

    # Add polynomial terms to distance matrix
    
    # Check if there are any polynomial kernels to add
    if polyweights:
        order = 0
        for weight, offset in zip(polyweights, polyoffsets):
            order += 1
            if not weight: continue
            # Add polynomial norms of training data
            Knew = XtXt + offset
            Knew = Knew ** order
            Knew = Knew * weight
            Dxx  += np.broadcast_to(Knew, (m, n))

            # Add polynomial norms of testing data
            Knew = XvXv + offset
            Knew = Knew ** order
            Knew = Knew * weight
            Knew = np.expand_dims(Knew, 1)
            Dxx += np.broadcast_to(Knew, (m, n))

            # Add polynomial inner products
            Knew = XvXt + offset
            Knew = Knew ** order
            Knew = Knew * weight
            Dxx  -= 2 * Knew

    # Add radial basis function norms to Gram matrix

    # Check if there are any rbf kernels to add
    if rbfweights:

        # Map the data to a matrix whose entry (i, j) is ||xi - xj||^2
        XvXt = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                xi = X_valid[i, :]
                xj = X_train[j, :]
                XvXt[i, j] = np.sum( (xi - xj) ** 2)

        for weight, width in zip(rbfweights, rbfwidths):
            if not weight: continue
            Knew = weight * np.exp(-XvXt / width)
            Dxx -= 2 * Knew

    return Dxx


