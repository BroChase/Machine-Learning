import numpy as np
from scipy.special import expit


class StochLogisticRegression:

    def sig_function(self, z):
        return expit(z)

    def sgd_logistic(self, x, y, n_epoch, alpha, lmda):
        # Initialize thetas with random
        theta = np.random.uniform(size=x.shape[1])
        m = x.shape[0]
        epoch = 0
        while epoch < n_epoch:
            # from the samples select a random row
            trial = np.random.randint(0, len(y))
            # calculate the sigmoid of the x row * thetas
            pred = self.sig_function(np.dot(x[trial, :], theta))
            # get the 'error' between the predicted and actual y
            error = pred - y[trial]
            # gradiant = (pred[i] - actual[i]) * x[i]
            gradient = np.dot(x[trial, :], error)
            theta = theta - alpha * (gradient - lmda/m * theta)
            epoch += 1
        return theta
