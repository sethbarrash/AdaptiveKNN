import pandas as pd
import numpy  as np
import SparseGaussianProcess as sgp

from datetime import datetime
from itertools import product
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

import learners as lrn
from sklearn.tree import DecisionTreeClassifier

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

    x0 = np.atleast_2d(X[0])
    x1 = np.atleast_2d(X[1])

    g = sgp.SparseGaussianProcess(x0, k0, h, sigma)
    g.update(x1, k1)
    ridx = np.arange(2, len(X))

    return g, ridx


def add_training_datum(g, X, y, idx, ridx):
    knew = fitk_spincom(X, y, idx)
    xnew = np.atleast_2d(X[idx])
    g.update(xnew, knew)

    idx_ = np.searchsorted(ridx, idx)
    ridx = np.delete(ridx, idx_)


def find_most_novel_training_datum(g, X, y, ridx):
    Xr     = np.atleast_2d(X[ridx])
    gamma  = g.novelty(Xr)
    idx = np.argmin(gamma)
    return idx


def add_most_novel_training_datum(g, X, y, ridx):
    idxnew = find_most_novel_training_datum(g, X, y, ridx)
    add_training_datum(g, X, y, idxnew, ridx)


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
    alpha, kmax = int(alpha), int(kmax)

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
    yhat = knr.predict( np.atleast_2d(Xtest) )
    errors = yhat.squeeze() - ytest.squeeze()
    t2 = datetime.now()

    trainTime = (t1 - t0).total_seconds()
    testTime  = (t2 - t1).total_seconds()

    return errors, trainTime, testTime



def spincomTrainTest(X, y, Xtest, ytest, hyperparams):
    t0 = datetime.now()

    m, h, sigma = hyperparams.values()
    g = generalizek_spincom(X, y, m, h, sigma)

    t1 = datetime.now()

    n = len(X)
    X = atleast_2d_T(X)

    ktest  = np.zeros_like(ytest, dtype = int)
    errors = np.zeros_like(ytest)
    for i in range(len(ytest)):
        xi = np.atleast_2d(Xtest[i])
        ki = g.predict(xi)
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

    return errors, ktest, trainTime, testTime



def adaknnTrainTest(X, y, Xtest, ytest, hyperparams):
    t0 = datetime.now()

    alpha, kmax = hyperparams.values()
    g = generalizek_adaknn(X, y, alpha, kmax)

    t1 = datetime.now()

    n = len(X)
    X = atleast_2d_T(X)

    ktest  = np.zeros_like(ytest, dtype = int)
    errors = np.zeros_like(ytest)
    for i in range(len(ytest)):
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

    return errors, ktest, trainTime, testTime



def ktreeTrainTest(X, y, Xtest, ytest, hyperparams):
    t0 = datetime.now()

    rho1, rho2, k, sigma = hyperparams.values()

    lz = lrn.IKNNzhang(np.atleast_2d(X), y, rho1, rho2, k, sigma)
    ktrain = lz.Ktrain
    dtc = DecisionTreeClassifier()
    dtc.fit(np.atleast_2d(X), ktrain)

    t1 = datetime.now()

    n = len(X)
    ktest  = dtc.predict(np.atleast_2d(Xtest)).astype(int)
    ktest[ktest < 1] = 1
    ktest[ktest > n] = n

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

    return errors, ktest, trainTime, testTime









###############################################################################
## Optimize hyperparameters
###############################################################################

def optimize_k(Xtrain, ytrain, Xvalid, yvalid, ks):
    knn_validation = pd.DataFrame({"error" : np.nan}, ks)
    for k in ks:
        errors = knnTrainTest(Xtrain, ytrain, Xvalid, yvalid, k)[0]
        knn_validation["error"][k] = np.mean( errors ** 2 )
    kbest = knn_validation["error"].idxmin()

    return kbest, knn_validation



def optimize_sigma(Xtrain, ytrain, Xvalid, yvalid, sigmas, m, h):
    spincom_validation = pd.DataFrame({"error" : np.nan}, sigmas)
    for sigma in sigmas:
        hyperparams = {
            "m"     : m,
            "h"     : h,
            "sigma" : sigma,
        }
        errors = spincomTrainTest(Xtrain, ytrain, Xvalid, yvalid, hyperparams)[0]
        spincom_validation["error"][sigma] = np.mean( errors ** 2 )
    sigmabest = spincom_validation["error"].idxmin()

    return sigmabest, spincom_validation



def alphas_kmaxs_to_index(alphas, kmaxs):
    n_alphas = len(alphas)
    n_kmaxs  = len(kmaxs)
    alphas_idx = np.repeat(alphas, n_kmaxs)
    kmaxs_idx  = np.tile(kmaxs, n_alphas)

    idx_keep = np.less_equal(alphas_idx, kmaxs_idx)
    alphas_idx = alphas_idx[idx_keep]
    kmaxs_idx  = kmaxs_idx[idx_keep]

    idx = pd.MultiIndex.from_arrays([alphas_idx, kmaxs_idx])
    return idx



def optimize_hyperparams_adaknn(Xtrain, ytrain, Xvalid, yvalid, alphas, kmaxs):
    idx = alphas_kmaxs_to_index(alphas, kmaxs)
    adaknn_validation = pd.DataFrame({"error" : np.nan}, idx)
    for alpha, kmax in idx.values:
        hyperparams = {
            "alpha" : alpha,
            "kmax"  : kmax,
        }
        errors = adaknnTrainTest(Xtrain, ytrain, Xvalid, yvalid, hyperparams)[0]
        adaknn_validation.loc[alpha, kmax]["error"] = np.sum(errors ** 2)
    alphabest, kmaxbest = adaknn_validation["error"].idxmin()

    return alphabest, kmaxbest, adaknn_validation

def optimize_hyperparams_ktree(Xtrain, ytrain, Xvalid, yvalid, hyperparam_grid):
    idx = pd.MultiIndex.from_product(hyperparam_grid.values())
    ktree_validation = pd.DataFrame({"error" : np.nan}, idx)
    for rho1, rho2, k, sigma in idx.values:
        hyperparams = {
            "rho1" : rho1,
            "rho2" : rho2,
            "k"    : k,
            "sigma" : sigma,
        }
        errors = ktreeTrainTest(Xtrain, ytrain, Xvalid, yvalid, hyperparams)[0]
        ktree_validation.loc[rho1, rho2, k, sigma]["error"] = np.sum(errors ** 2)
    rho1best, rho2best, kbest, sigmabest = ktree_validation["error"].idxmin()

    return rho1best, rho2best, kbest, sigmabest, ktree_validation







###############################################################################
## Real data trials
###############################################################################

def real_data_trial_knn(data_partition, ks):
    dataset, Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest = data_partition
    t0 = datetime.now()
    k  = optimize_k(Xtrain, ytrain, Xvalid, yvalid, ks)[0]
    t1 = datetime.now()
    hyperparameterTime = (t1 - t0).total_seconds()

    X = np.vstack((Xtrain, Xvalid))
    y = np.append(ytrain, yvalid)
    errors, trainTime, testTime = knnTrainTest(X, y, Xtest, ytest, k)
    error = np.sum(errors ** 2)

    new_row = pd.Series({
        "dataset" : dataset,
        "k"       : k,
        "error"   : error,
        "hyperparameterTime" : hyperparameterTime,
        "trainTime"          : trainTime,
        "testTime"           : testTime,
    })
    return new_row

def real_data_trial_spincom(data_partition, sigmas, m, h):
    dataset, Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest = data_partition
    t0 = datetime.now()
    sigma  = optimize_sigma(Xtrain, ytrain, Xvalid, yvalid, sigmas, m, h)[0]
    t1 = datetime.now()
    hyperparameterTime = (t1 - t0).total_seconds()

    X = np.vstack((Xtrain, Xvalid))
    y = np.append(ytrain, yvalid)
    hyperparams = {"m" : m, "h" : h, "sigma" : sigma}
    errors, ktest, trainTime, testTime = spincomTrainTest(X, y, Xtest, ytest, hyperparams)
    error = np.sum(errors ** 2)

    new_row = pd.Series({
        "dataset" : dataset,
        "sigma"   : sigma,
        "m"       : m,
        "h"       : h,
        "error"   : error,
        "hyperparameterTime" : hyperparameterTime,
        "trainTime"          : trainTime,
        "testTime"           : testTime,
    })
    return new_row

def real_data_trial_adaknn(data_partition, alphas, kmaxs):
    dataset, Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest = data_partition
    t0 = datetime.now()
    alpha, kmax  = optimize_hyperparams_adaknn(Xtrain, ytrain, Xvalid, yvalid, alphas, kmaxs)[:2]
    t1 = datetime.now()
    hyperparameterTime = (t1 - t0).total_seconds()

    X = np.vstack((Xtrain, Xvalid))
    y = np.append(ytrain, yvalid)
    hyperparams = {"alpha" : alpha, "kmax" : kmax}
    errors, ktest, trainTime, testTime = adaknnTrainTest(X, y, Xtest, ytest, hyperparams)
    error = np.sum(errors ** 2)

    new_row = pd.Series({
        "dataset" : dataset,
        "alpha"   : alpha,
        "kmax"    : kmax,
        "error"   : error,
        "hyperparameterTime" : hyperparameterTime,
        "trainTime"          : trainTime,
        "testTime"           : testTime,
    })
    return new_row

def real_data_trial_ktree(data_partition, hyperparam_grid):
    dataset, Xtrain, Xvalid, Xtest, ytrain, yvalid, ytest = data_partition
    t0 = datetime.now()
    rho1, rho2, k, sigma = \
         optimize_hyperparams_ktree(Xtrain, ytrain, Xvalid, yvalid, hyperparam_grid)[:4]
    t1 = datetime.now()
    hyperparameterTime = (t1 - t0).total_seconds()

    X = np.vstack((Xtrain, Xvalid))
    y = np.append(ytrain, yvalid)
    hyperparams = {
        "rho1"  : rho1,
        "rho2"  : rho2,
        "k"     : k,
        "sigma" : sigma,
    }
    errors, ktest, trainTime, testTime = ktreeTrainTest(X, y, Xtest, ytest, hyperparams)
    error = np.sum(errors ** 2)

    new_row = pd.Series({
        "dataset" : dataset,
        "rho1"    : rho1,
        "rho2"    : rho2,
        "k"       : k,
        "sigma"   : sigma,
        "error"   : error,
        "hyperparameterTime" : hyperparameterTime,
        "trainTime"          : trainTime,
        "testTime"           : testTime,
    })
    return new_row