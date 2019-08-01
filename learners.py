import numpy  as np
import pandas as pd
import optimization as opt
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier

class IKNNlinear:

    def __init__(self, Xtrain, ytrain, rho, c = 0.001, tol = 0.001, thresh = 0.001):

        # Optimization parameters
        self.rho    = rho
        self.c      = c
        self.tol    = tol
        self.thresh = thresh

        # Decision tree
        self.tree = DecisionTreeClassifier()

        # Train IkNN learner
        t0        = datetime.now()
        N         = Xtrain.shape[0]
        Kxx       = Xtrain @ Xtrain.T
        Ainv      = np.linalg.inv( Kxx + 0.5 * c * np.eye(N) )
        W, conv, div, itr = opt.admm(Kxx, Ainv, rho, c, tol, thresh)
        Ktrain    = np.sum(W > 0, 1)
        density   = np.sum(Ktrain) / N ** 2

        t1        = datetime.now()
        trainTime = (t1 - t0).total_seconds()

        self.tree.fit(Xtrain, Ktrain)

        # Training results
        self.W         = W
        self.Ktrain    = Ktrain
        self.density   = density
        self.trainTime = trainTime
        self.convergence = int(conv)
        self.divergence  = int(div)
        self.iterations  = itr

        # Testing results
        self.Ktest    = np.nan
        self.correct  = np.nan
        self.accuracy = np.nan
        self.testTime = np.nan

    def grad_smooth(self, Xtrain):
        """
        Compute the gradient of the objective of this learner at the value of 
        W it has learned.

        If optimization worked correctly, this gradient should (approximately)
        satisfy the optimality conditions of the objective.
        """

        N = Xtrain.shape[0]
        XX   = Xtrain @ Xtrain.T
        grad = 2 * XX * (self.W - np.eye(N))
        return grad

    def checkOptimality(self, Xtrain, tol):
        grad_smooth = self.grad_smooth(Xtrain)

        ## Check that the entries on the diagonal are zero
        cond_diag = np.any( np.diag(self.W) )
        ## Check that the zero off-diagonal entries of W correspond to a nonnegative gradient
        idx_zero  = np.logical_not(self.W)
        np.fill_diagonal(idx_zero, False)
        cond_zero = grad_smooth[idx_zero] + self.rho > tol
        ## Check that the positive off-diagonal entries of W correspond to a zero gradient
        idx_pos   = self.W > 0
        np.fill_diagonal(idx_pos,  False)
        cond_pos  = -1 * grad_smooth[idx_pos] > tol and grad_smooth[idx_pos] < tol

        verdad = cond_diag and cond_zero and cond_pos

        return verdad

    def test(self, Xtest, ytest, pred):
        """
        Compute the numbers of neighbors to be used for each testing datum.
        Classify that datum according to majority vote.
        Evaluate classification accuracy.
        """
        t0 = datetime.now()

        Ktest    = self.tree.predict(Xtest)
        Ktest[np.logical_not(Ktest)] = 1
        yhat     = pred[np.arange(len(Xtest)), Ktest - 1]
        correct  = np.equal(yhat, ytest.squeeze() )
        accuracy = np.mean(correct)

        t1 = datetime.now()
        testTime = (t1 - t0).total_seconds()

        self.Ktest    = Ktest
        self.correct  = correct
        self.accuracy = accuracy
        self.testTime = testTime

    def breakdown(self):
        """
        Map a trained IKNN object to a dataframe containing the occurences of
        choosing each value of K was chosen and the number of correct
        classifications among those occurences.
        """
        neighbor_counts = pd.Index( self.Ktest ).value_counts()
        nmax            = neighbor_counts.index.max()
        neighbors_idx   = np.arange(nmax + 1)
        frequency       = neighbor_counts.reindex(neighbors_idx, fill_value = 0)

        incorrect_idx = np.logical_not(self.correct.values)
        Ktest_correct = self.Ktest.copy()
        Ktest_correct[incorrect_idx] = np.nan
        k = Ktest_correct.value_counts().reindex(neighbors_idx, fill_value = 0)

        Breakdown = pd.DataFrame(
            index = neighbors_idx, 
            data = {"frequency": frequency, "correct": k}
        )

        return Breakdown



class IKNNsupervised:

    def __init__(self, Xtrain, ytrain, rho1, rho2, c = 0.001, tol = 0.001, thresh = 0.0):

        # Optimization parameters
        self.rho1   = rho1
        self.rho2   = rho2
        self.c      = c
        self.tol    = tol
        self.thresh = thresh

        # Decision tree
        self.tree = DecisionTreeClassifier()

        # Train IkNN learner
        t0        = datetime.now()
        N         = Xtrain.shape[0]
        L         = opt.computeLaplacian(ytrain)
        Kxx       = Xtrain @ Xtrain.T
        Ainv      = np.linalg.inv( Kxx + rho2 * L + 0.5 * c * np.eye(N) )
        W, conv, div, itr = opt.admm(Kxx, Ainv, rho1, c, tol, thresh)
        Ktrain    = np.sum(W > 0, 1)
        density   = np.sum(Ktrain) / N ** 2

        t1        = datetime.now()
        trainTime = (t1 - t0).total_seconds()

        self.tree.fit(Xtrain, Ktrain)

        # Training results
        self.W          = W
        self.Ktrain     = Ktrain
        self.density    = density
        self.trainTime  = trainTime
        self.convergence = int(conv)
        self.divergence  = int(div)
        self.iterations  = itr

        # Testing results
        self.Ktest    = np.nan
        self.correct  = np.nan
        self.accuracy = np.nan
        self.testTime = np.nan

    def grad_smooth(self, Xtrain, ytrain):
        """
        Compute the gradient of the objective of this learner at the value of 
        W it has learned.

        If optimization worked correctly, this gradient should (approximately)
        satisfy the optimality conditions of the objective.
        """

        L    = opt.computeLaplacian(ytrain)
        XX   = Xtrain @ Xtrain.T
        grad = 2 * (XX - self.rho2 * L) @ self.W - 2 * XX
        return grad

    def checkOptimality(self, Xtrain, ytrain, tol):
        grad_smooth = self.grad_smooth(Xtrain, ytrain)

        ## Check that the entries on the diagonal are zero
        cond_diag = np.any( np.diag(self.W) )
        ## Check that the zero off-diagonal entries of W correspond to a nonnegative gradient
        idx_zero  = np.logical_not(self.W)
        np.fill_diagonal(idx_zero, False)
        cond_zero = grad_smooth[idx_zero] + self.rho1 > tol
        ## Check that the positive off-diagonal entries of W correspond to a zero gradient
        idx_pos   = self.W > 0
        np.fill_diagonal(idx_pos,  False)
        cond_pos  = -1 * grad_smooth[idx_pos] > tol and grad_smooth[idx_pos] < tol

        verdad = cond_diag and cond_zero and cond_pos

        return verdad

    def test(self, Xtest, ytest, pred):
        """
        Compute the numbers of neighbors to be used for each testing datum.
        Classify that datum according to majority vote.
        Evaluate classification accuracy.
        """
        t0 = datetime.now()

        Ktest    = self.tree.predict(Xtest)
        Ktest[np.logical_not(Ktest)] = 1
        yhat     = pred[np.arange(len(Xtest)), Ktest - 1]
        correct  = np.equal(yhat, ytest.squeeze() )
        accuracy = np.mean(correct)

        t1 = datetime.now()
        testTime = (t1 - t0).total_seconds()

        self.Ktest    = Ktest
        self.correct  = correct
        self.accuracy = accuracy
        self.testTime = testTime

    def breakdown(self):
        """
        Map a trained IKNN object to a dataframe containing the occurences of
        choosing each value of K was chosen and the number of correct
        classifications among those occurences.
        """
        neighbor_counts = pd.Index( self.Ktest ).value_counts()
        nmax            = neighbor_counts.index.max()
        neighbors_idx   = np.arange(nmax + 1)
        frequency       = neighbor_counts.reindex(neighbors_idx, fill_value = 0)

        incorrect_idx = np.logical_not(self.correct.values)
        Ktest_correct = self.Ktest.copy()
        Ktest_correct[incorrect_idx] = np.nan
        k = Ktest_correct.value_counts().reindex(neighbors_idx, fill_value = 0)

        Breakdown = pd.DataFrame(
            index = neighbors_idx, 
            data = {"frequency": frequency, "correct": k}
        )

        return Breakdown



class IKNNsupervisedLocal:

    def __init__(self, Xtrain, ytrain, rho1, rho2, k, c = 0.001, tol = 0.001, thresh = 0.0):

        # Optimization parameters
        self.rho1   = rho1
        self.rho2   = rho2
        self.k      = k
        self.c      = c
        self.tol    = tol
        self.thresh = thresh

        # Decision tree
        self.tree = DecisionTreeClassifier()

        # Train IkNN learner
        t0        = datetime.now()
        N         = Xtrain.shape[0]
        L         = opt.computeLaplacianLocal(Xtrain, ytrain, k)
        Kxx       = Xtrain @ Xtrain.T
        Ainv      = np.linalg.inv( Kxx + rho2 * L + 0.5 * c * np.eye(N) )
        W, conv, div, itr = opt.admm(Kxx, Ainv, rho1, c, tol, thresh)
        Ktrain    = np.sum(W > 0, 1)
        density   = np.sum(Ktrain) / N ** 2

        t1        = datetime.now()
        trainTime = (t1 - t0).total_seconds()

        self.tree.fit(Xtrain, Ktrain)

        # Training results
        self.W          = W
        self.Ktrain     = Ktrain
        self.density    = density
        self.trainTime  = trainTime
        self.convergence = int(conv)
        self.divergence  = int(div)
        self.iterations  = itr

        # Testing results
        self.Ktest    = np.nan
        self.correct  = np.nan
        self.accuracy = np.nan
        self.testTime = np.nan

    def test(self, Xtest, ytest, pred):
        """
        Compute the numbers of neighbors to be used for each testing datum.
        Classify that datum according to majority vote.
        Evaluate classification accuracy.
        """
        t0 = datetime.now()

        Ktest    = self.tree.predict(Xtest)
        Ktest[np.logical_not(Ktest)] = 1
        yhat     = pred[np.arange(len(Xtest)), Ktest - 1]
        correct  = np.equal(yhat, ytest.squeeze() )
        accuracy = np.mean(correct)

        t1 = datetime.now()
        testTime = (t1 - t0).total_seconds()

        self.Ktest    = Ktest
        self.correct  = correct
        self.accuracy = accuracy
        self.testTime = testTime\

    def grad_smooth(self, Xtrain, ytrain):
        """
        Compute the gradient of the objective of this learner at the value of 
        W it has learned.

        If optimization worked correctly, this gradient should (approximately)
        satisfy the optimality conditions of the objective.
        """

        L    = opt.computeLaplacianLocal(Xtrain, ytrain, self.k)
        XX   = Xtrain @ Xtrain.T
        grad = 2 * (XX - self.rho2 * L) @ self.W - 2 * XX
        return grad

    def checkOptimality(self, Xtrain, ytrain, tol):
        grad_smooth = self.grad_smooth(Xtrain, ytrain)

        ## Check that the entries on the diagonal are zero
        cond_diag = np.any( np.diag(self.W) )
        ## Check that the zero off-diagonal entries of W correspond to a nonnegative gradient
        idx_zero  = np.logical_not(self.W)
        np.fill_diagonal(idx_zero, False)
        cond_zero = grad_smooth[idx_zero] + self.rho1 > tol
        ## Check that the positive off-diagonal entries of W correspond to a zero gradient
        idx_pos   = self.W > 0
        np.fill_diagonal(idx_pos,  False)
        cond_pos  = -1 * grad_smooth[idx_pos] > tol and grad_smooth[idx_pos] < tol

        verdad = cond_diag and cond_zero and cond_pos

        return verdad

    def breakdown(self):
        """
        Map a trained IKNN object to a dataframe containing the occurences of
        choosing each value of K was chosen and the number of correct
        classifications among those occurences.
        """
        neighbor_counts = pd.Index( self.Ktest ).value_counts()
        nmax            = neighbor_counts.index.max()
        neighbors_idx   = np.arange(nmax + 1)
        frequency       = neighbor_counts.reindex(neighbors_idx, fill_value = 0)

        incorrect_idx = np.logical_not(self.correct.values)
        Ktest_correct = self.Ktest.copy()
        Ktest_correct[incorrect_idx] = np.nan
        k = Ktest_correct.value_counts().reindex(neighbors_idx, fill_value = 0)

        Breakdown = pd.DataFrame(
            index = neighbors_idx, 
            data = {"frequency": frequency, "correct": k}
        )

        return Breakdown







class IKNNzhang:

    def __init__(self, Xtrain, ytrain, rho1, rho2, k = 5, sigma = 0.1, c = 0.001, tol = 0.001, thresh = 0.0):

        # Optimization parameters
        self.rho1   = rho1
        self.rho2   = rho2
        self.k      = k
        self.sigma  = sigma
        self.c      = c
        self.tol    = tol
        self.thresh = thresh

        # Decision tree
        self.tree = DecisionTreeClassifier()

        # Train IkNN learner
        t0        = datetime.now()
        L         = opt.laplacianZhang(Xtrain, k, sigma)
        N, D      = Xtrain.shape
        Kxx       = Xtrain @ Xtrain.T
        Ainv      = np.linalg.inv( Xtrain @ (np.eye(D) + rho2 * L) @ Xtrain.T + 0.5 * c * np.eye(N))
        W, conv, div, itr = opt.admm(Kxx, Ainv, rho1, c, tol, thresh)
        Ktrain    = np.sum(W > 0, 1)
        density   = np.sum(Ktrain) / N ** 2
        t1        = datetime.now()
        trainTime = (t1 - t0).total_seconds()

        self.tree.fit(Xtrain, Ktrain)

        # Training results
        self.W         = W
        self.Ktrain    = Ktrain
        self.density   = density
        self.trainTime = trainTime
        self.convergence = int(conv)
        self.divergence  = int(div)
        self.iterations  = itr

        # Testing results
        self.Ktest    = np.nan
        self.correct  = np.nan
        self.accuracy = np.nan
        self.testTime = np.nan

    def test(self, Xtest, ytest, pred):
        """
        Compute the numbers of neighbors to be used for each testing datum.
        Classify that datum according to majority vote.
        Evaluate classification accuracy.
        """
        t0 = datetime.now()

        Ktest    = self.tree.predict(Xtest)
        Ktest[np.logical_not(Ktest)] = 1
        yhat     = pred[np.arange(len(Xtest)), Ktest - 1]
        correct  = np.equal(yhat, ytest.squeeze() )
        accuracy = np.mean(correct)

        t1 = datetime.now()
        testTime = (t1 - t0).total_seconds()

        self.Ktest    = Ktest
        self.correct  = correct
        self.accuracy = accuracy
        self.testTime = testTime

    def grad_smooth(self, Xtrain):
        """
        Compute the gradient of the objective of this learner at the value of 
        W it has learned.

        If optimization worked correctly, this gradient should (approximately)
        satisfy the optimality conditions of the objective.
        """

        L    = opt.laplacianZhang(Xtrain, self.k, self.sigma)
        XX   = Xtrain @ Xtrain.T
        grad = 2 * (XX - self.rho2 * Xtrain @ L @ Xtrain.T) @ self.W - 2 * XX
        return grad

    def checkOptimality(self, Xtrain, tol):

        grad_smooth = self.grad_smooth(Xtrain)
        ## Check that the zero off-diagonal entries of W correspond to a nonnegative gradient
        idx_zero  = np.logical_not(self.W)
        cond_zero = grad_smooth[idx_zero] + self.rho1 > tol
        ## Check that the positive off-diagonal entries of W correspond to a zero gradient
        idx_pos   = self.W > 0
        cond_pos  = -1 * grad_smooth[idx_pos] > tol and grad_smooth[idx_pos] < tol

        verdad = cond_zero and cond_pos

        return verdad

    def breakdown(self):
        """
        Map a trained IKNN object to a dataframe containing the occurences of
        choosing each value of K was chosen and the number of correct
        classifications among those occurences.
        """
        neighbor_counts = pd.Index( self.Ktest ).value_counts()
        nmax            = neighbor_counts.index.max()
        neighbors_idx   = np.arange(nmax + 1)
        frequency       = neighbor_counts.reindex(neighbors_idx, fill_value = 0)

        incorrect_idx = np.logical_not(self.correct.values)
        Ktest_correct = self.Ktest.copy()
        Ktest_correct[incorrect_idx] = np.nan
        k = Ktest_correct.value_counts().reindex(neighbors_idx, fill_value = 0)

        Breakdown = pd.DataFrame(
            index = neighbors_idx, 
            data = {"frequency": frequency, "correct": k}
        )

        return Breakdown








class IKNNkernel:

    def __init__(self, Xtrain, ytrain, Xtest, ytest, rho, polyweights = [],
    polyoffsets = [], rbfweights = [], rbfwidths = [], c = 0.001, tol = 0.001,
    thresh = 0.0):

        # Optimization parameters
        self.rho    = rho
        self.c      = c
        self.tol    = tol
        self.thresh = thresh

        # Decision tree
        self.tree = DecisionTreeClassifier()

        # Train IkNN learner
        t0        = datetime.now()
        N      = Xtrain.shape[0]
        Kxx    = opt.gramMatrix(Xtrain, polyweights, polyoffsets, rbfweights, rbfwidths)
        Ainv   = np.linalg.inv( Kxx + 0.5 * c * np.eye(Xtrain.shape[0]))
        W, conv, div, itr = opt.admm(Kxx, Ainv, rho, c, tol, thresh)
        Ktrain = np.sum(W > 0, 1)
        density   = np.sum(Ktrain) / N ** 2

        t1        = datetime.now()
        trainTime = (t1 - t0).total_seconds()

        self.tree.fit(Xtrain, Ktrain)

        # Training results
        self.W         = W
        self.density   = density
        self.Ktrain    = Ktrain
        self.trainTime = trainTime
        self.convergence = int(conv)
        self.divergence  = int(div)
        self.iterations  = itr

        # Testing results
        self.Ktest    = np.nan
        self.correct  = np.nan
        self.accuracy = np.nan
        self.testTime = np.nan

    # def test(self, Xtest, ytest, pred):
    #     """
    #     Compute the numbers of neighbors to be used for each testing datum.
    #     Classify that datum according to majority vote.
    #     Evaluate classification accuracy.
    #     """
        t0 = datetime.now()

        Ktest    = self.tree.predict(Xtest)
        Ktest[np.logical_not(Ktest)] = 1
        Dxx      = opt.distancesKer(Xtrain, Xtest, polyweights, polyoffsets, rbfweights, rbfwidths)
        yhat     = opt.predictKer(Ktest, Dxx, ytrain)
        correct  = np.equal(yhat, ytest.squeeze() )
        accuracy = np.mean(correct)

        t1 = datetime.now()
        testTime = (t1 - t0).total_seconds()

        self.Ktest    = Ktest
        self.correct  = correct
        self.accuracy = accuracy
        self.testTime = testTime

    def breakdown(self):
        """
        Map a trained IKNN object to a dataframe containing the occurences of
        choosing each value of K was chosen and the number of correct
        classifications among those occurences.
        """
        neighbor_counts = pd.Index( self.Ktest ).value_counts()
        nmax            = neighbor_counts.index.max()
        neighbors_idx   = np.arange(nmax + 1)
        frequency       = neighbor_counts.reindex(neighbors_idx, fill_value = 0)

        incorrect_idx = np.logical_not(self.correct.values)
        Ktest_correct = self.Ktest.copy()
        Ktest_correct[incorrect_idx] = np.nan
        k = Ktest_correct.value_counts().reindex(neighbors_idx, fill_value = 0)

        Breakdown = pd.DataFrame(
            index = neighbors_idx, 
            data = {"frequency": frequency, "correct": k}
        )

        return Breakdown




class IKNNkernelSupervised:

    def __init__(self, Xtrain, ytrain, Xtest, ytest, rho1, rho2, 
    polyweights = [], polyoffsets = [], rbfweights = [], rbfwidths = [],
    c = 0.001, tol = 0.001, thresh = 0.0):

        # Optimization parameters
        self.rho1   = rho1
        self.rho2   = rho2
        self.c      = c
        self.tol    = tol
        self.thresh = thresh

        # Decision tree
        self.tree = DecisionTreeClassifier()

        # Train IkNN learner
        t0        = datetime.now()
        N      = Xtrain.shape[0]
        Kxx    = opt.gramMatrix(Xtrain, polyweights, polyoffsets, rbfweights, rbfwidths)
        L      = opt.computeLaplacian(ytrain)
        Ainv   = np.linalg.inv( Kxx + rho2 * L + 0.5 * c * np.eye(N) )
        W, conv, div, itr = opt.admm(Kxx, Ainv, rho1, c, tol, thresh)
        Ktrain = np.sum(W > 0, 1)
        density   = np.sum(Ktrain) / N ** 2

        t1        = datetime.now()
        trainTime = (t1 - t0).total_seconds()

        self.tree.fit(Xtrain, Ktrain)

        # Training results
        self.W         = W
        self.Ktrain    = Ktrain
        self.density   = density
        self.trainTime = trainTime
        self.convergence = int(conv)
        self.divergence  = int(div)
        self.iterations  = itr

        # Testing results
        self.Ktest    = np.nan
        self.correct  = np.nan
        self.accuracy = np.nan
        self.testTime = np.nan

    # def test(self, Xtest, ytest, pred):
    #     """
    #     Compute the numbers of neighbors to be used for each testing datum.
    #     Classify that datum according to majority vote.
    #     Evaluate classification accuracy.
    #     """
        t0 = datetime.now()

        Ktest    = self.tree.predict(Xtest)
        Ktest[np.logical_not(Ktest)] = 1
        Dxx      = opt.distancesKer(Xtrain, Xtest, polyweights, polyoffsets, rbfweights, rbfwidths)
        yhat     = opt.predictKer(Ktest, Dxx, ytrain)
        correct  = np.equal(yhat, ytest.squeeze() )
        accuracy = np.mean(correct)

        t1 = datetime.now()
        testTime = (t1 - t0).total_seconds()

        self.Ktest    = Ktest
        self.correct  = correct
        self.accuracy = accuracy
        self.testTime = testTime

    def breakdown(self):
        """
        Map a trained IKNN object to a dataframe containing the occurences of
        choosing each value of K was chosen and the number of correct
        classifications among those occurences.
        """
        neighbor_counts = pd.Index( self.Ktest ).value_counts()
        nmax            = neighbor_counts.index.max()
        neighbors_idx   = np.arange(nmax + 1)
        frequency       = neighbor_counts.reindex(neighbors_idx, fill_value = 0)

        incorrect_idx = np.logical_not(self.correct.values)
        Ktest_correct = self.Ktest.copy()
        Ktest_correct[incorrect_idx] = np.nan
        k = Ktest_correct.value_counts().reindex(neighbors_idx, fill_value = 0)

        Breakdown = pd.DataFrame(
            index = neighbors_idx, 
            data = {"frequency": frequency, "correct": k}
        )

        return Breakdown




def IKNN(Xtrain, ytrain, Xtest, ytest, algorithm, hyperparams, c, tol, thresh):
    """
    Inputs:
    Xtrain - training data
    ytrain - trianing labels
    Xtest  - testing data
    ytest  - testing labels
    algorithm   - variant of IKNN to use
    hyperparams - hyperparameters of this IKNN variant in list form
    c      - ADMM stepsize
    tol    - tolerance for stopping criterion
    thresh - for getting rid of insignificant entries in final W

    Ouptput:
    Trained and tested learner
    """

    ###########################################################################
    # Non-kernelized learners suffer from the fact that they must retrieve
    # predictions by kNN. I have not yet written code to retrieve these.
    ###########################################################################

    # if   algorithm == "linear" :
    #     rho = hyperparams[0]
    #     learner = IKNNlinear(Xtrain, ytrain, rho, c, tol, thresh)
    #     return learner

    # elif algorithm == "linearSupervised" :
    #     rho1, rho2 = hyperparameters
    #     learner = IKNNsupervised()
    #     pass

    # elif algorithm == "square" :
    #     pass

    # elif algorithm == "squareSupervised" :
    #     pass

    # elif algorithm == "cubic" :
    #     pass

    # elif algorithm == "cubicSupervised" :
    #     pass

    # elif algorithm == "rbf" :
    #     pass

    # elif algorithm == "rbfSupervised" :
    #     pass

    # elif algorithm == "zhang" :
    #     pass
