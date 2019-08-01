import numpy as np
import SparseGP as sgp

from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

def get_neighbor_order(X, y, idx):
    x0 = X[idx]
    if len(X.shape) == 1 : xdist = np.abs(X - x0)
    else                 : xdist = np.abs( np.sum(X - x0, 1) )
    neighbor_order = np.argsort(xdist)[1:]

    return neighbor_order


def get_neighborhood(X, y, idx):
    nidx = get_neighbor_order(X, y, idx)

    Xsorted = X[nidx]
    ysorted = y[nidx]

    return Xsorted, ysorted


def calculate_neighbor_errors(ysorted, y0):
    ysum = np.cumsum(ysorted)
    ybar = ysum / np.arange(1, len(ysorted) + 1)
    e = ybar - y0
    return e


def fitk_spincom_incorrect(X, y, idx):
    y0 = y[idx]
    Xsorted, ysorted = get_neighborhood(X, y, idx)
    e = calculate_neighbor_errors(ysorted, y0)
    k = np.argmin(e) + 1

    return k


def fitk_spincom(X, y, idx):
    y0 = y[idx]
    Xsorted, ysorted = get_neighborhood(X, y, idx)
    e = calculate_neighbor_errors(ysorted, y0)
    k = np.argmin(np.abs(e)) + 1

    return k


def initialize_k_generalization(X, y, h, sigma):
    k0 = fitk_spincom(X, y, 0)
    k1 = fitk_spincom(X, y, 1)

    g = sgp.SparseGP(X[0], k0, h, sigma)
    g.update(X[1], k1)
    ridx = np.arange(2, len(X))

    return g, ridx


def add_most_novel_training_datum(g, X, y, ridx):
    gamma = g.novelty(X, ridx)
    idxnew = np.argmin(gamma)
    knew = fitk_spincom(X, y, idxnew)
    g.update(X[idxnew], knew)
    idx_ = np.searchsorted(ridx, idxnew)
    ridx = np.delete(ridx, idx_)


def generalizek_adaknn(X, y, alpha, kmax):
    ktrain = np.zeros_like(y)
    for i in range(len(X)):
        ktrain[i] = fitk_adaknn(X, y, i, alpha, kmax)

    X = atleast_2d_T(X)
    mlp = MLPRegressor([15])
    mlp.fit(X, ktrain)

    return mlp


def generalizek_spincom(X, y, m, h, sigma):
    g, ridx = initialize_k_generalization(X, y, h, sigma)
    for _ in range(2, m):
        add_most_novel_training_datum(g, X, y, ridx)

    return g


def choose_k_from_random_subset(Krand, ysorted, y0):
    error_min = np.Inf
    for k in Krand:
        error = np.mean(ysorted[:k]) - y0
        if error < error_min:
            error_min = error
            kbest = k

    return kbest


def fitk_adaknn(X, y, idx, alpha, kmax):
    y0 = y[idx]
    Xsorted, ysorted = get_neighborhood(X, y, idx)
    Krand = np.random.choice(np.arange(kmax), alpha)
    kbest = choose_k_from_random_subset(Krand, ysorted, y0)
    return kbest


def chisel_k(ki, n):
    k = int(ki)
    if k < 1 : k = 1
    if k > n : k = n

    return k


def atleast_2d_T(X):
    try:
        if len(X.shape) == 1: return np.expand_dims(X, 1)
        else : return X
    except(AttributeError):
        return np.atleast_2d(X)


###############################################################################
## Train/test
###############################################################################

def knnTrainTest(X, y, Xtest, ytest, ki):
    knr  = KNeighborsRegressor(n_neighbors = ki, algorithm = "brute")
    t0 = datetime.now()
    knr.fit(X, y)
    t1 = datetime.now()
    ## If this receives multiple one-dimensional testing data, it will 
    ## misinterpret them as a single, multi-dimensional datum
    yhat = knr.predict( np.atleast_2d(Xtest) )[0]
    error = yhat - ytest
    t2 = datetime.now()

    trainTime = (t1 - t0).total_seconds()
    testTime  = (t2 - t1).total_seconds()

    return error, trainTime, testTime


def spincomTrainTest(X, y, Xtest, ytest, hyperparams):
    t0 = datetime.now()

    m, h, sigma = hyperparams.values()
    g = generalizek_spincom(X, y, m, h, sigma)

    t1 = datetime.now()

    n = len(X)
    X = atleast_2d_T(X)

    ktest  = np.zeros_like(y)
    errors = np.zeros_like(y)
    for i in range(len(y)):
        xi = Xtest[i]
        ki = g.a(xi)
        ki = chisel_k(ki, n)
        ktest[i] = ki

        error = knnTrainTest(X, y, xi, ytest[i], ki)
        try:
            errors[i] = error[0]
        except(IndexError):
            errors[i] = error

    t2 = datetime.now()

    trainTime = (t1 - t0).total_seconds()
    testTime  = (t2 - t1).total_seconds()

    return ktest, errors, trainTime, testTime


def adaknnTrainTest(X, y, Xtest, ytest, hyperparams):
    t0 = datetime.now()

    alpha, kmax = hyperparams.values()
    g = generalizek_adaknn(X, y, alpha, kmax)

    t1 = datetime.now()

    n = len(X)
    X = atleast_2d_T(X)

    ktest  = np.zeros_like(y)
    errors = np.zeros_like(y)
    for i in range(len(y)):
        xi = Xtest[i]
        ki = g.predict(np.atleast_2d(xi))
        ki = chisel_k(ki, n)
        ktest[i] = ki

        error = knnTrainTest(X, y, xi, ytest[i], ki)
        try:
            errors[i] = error[0]
        except(IndexError):
            errors[i] = error

    t2 = datetime.now()

    trainTime = (t1 - t0).total_seconds()
    testTime  = (t2 - t1).total_seconds()

    return ktest, errors, trainTime, testTime




def ktreeTrainTest(X, y, Xtest, ytest, hyperparams):
    t0 = datetime.now()

    lz = lrn.IKNNzhang(np.atleast_2d(X), y, rho1, rho2, k, sigma)
    ktrain = lz.Ktrain
    dtc = DecisionTreeClassifier()
    dtc.fit(np.atleast_2d(X), ktrain)

    t1 = datetime.now()

    ktest  = dtc.predict(np.atleast_2d(Xtest))
    ktest = int(ktest)
    if ktest < 1: ktest = 1
    if ktest > n: ktest = n

    errors = np.zeros_like(ytest)
    for i in range(len(ytest)):
        xi = Xtest[i]
        ki = ktest[i]

        error = knnTrainTest(X, y, xi, ytest[i], ki)
        try:
            errors[i] = error[0]
        except(IndexError):
            errors[i] = error

    t2 = datetime.now()

    trainTime = (t1 - t0).total_seconds()
    testTime  = (t2 - t1).total_seconds()

    return ktest, errors, trainTime, testTime









###############################################################################
## Optimize hyperparameters
###############################################################################

def optimize_k(Xtrain, ytrain, Xvalid, yvalid, ks):
    validation_results = pd.DataFrame(index = ks, {"error" : np.nan})
    for k in ks:
        knnTrainTest(Xtrain, ytrain, Xvalid, yvalid, k)