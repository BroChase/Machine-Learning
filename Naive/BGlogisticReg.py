import numpy as np
from scipy.special import expit


class LogisticRegression:

    def sig_function(self, z):
        return expit(z)

    def regBatchGD(self, alpha, x, y, n_epoch, lam):
        m = x.shape[0]  # number of samples
        theta = np.random.uniform(size=x.shape[1])
        for iter in range(0, n_epoch):
            hypothesis = self.sig_function(x.dot(theta))
            error = hypothesis - y
            gradient = np.dot(x.T, error)
            theta = theta - alpha * (gradient - (lam / m) * theta)   # update the theta for for next loop
            # theta = theta - (alpha * gradient + (lam / m) * theta)
        return theta

    def tolconv(self, y_pred, tol):
        for index, item in enumerate(y_pred):
            s = self.sig_function(item)
            if s >= tol:
                y_pred[index] = 1
            elif s < tol:
                y_pred[index] = 0
        return y_pred

    def y_pred(self, x_test, thetas):
        y_pred = x_test.dot(thetas)
        return y_pred