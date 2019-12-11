import attr
import numpy as np
from scipy.optimize import curve_fit
from optimistic.models import Model

@attr.s
class Gaussian(Model):
    def gaussian(self, X, *args):
        ''' Args:
                X: an N-element array representing a generally multidimensional point
                args:  first element: amplitude; next N: center; next N: width
        '''
        X = np.atleast_2d(X)
        N = X.shape[1]
        A = args[0]
        X0 = args[1:N+1]
        sigma = args[N+1:2*N+1]
        result = A
        for i in range(X.shape[1]):
            result *= np.exp(-(X[:,i]-X0[i])**2/sigma[i]**2)
        return result

    def fit(self, data):
        self.data = data
        N = len(self.parameters)
        p0 = tuple([0.5]*(2*N+1))
        points = data[self.parameters].values
        costs = data[self.experiment.__name__].values
        self.popt, self.pcov = curve_fit(self.gaussian, points, costs, p0)

    def predict(self, X):
        return self.gaussian(X, *tuple(self.popt)), 0
