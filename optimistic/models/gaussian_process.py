import attr
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from optimistic.models import Model

@attr.s
class GaussianProcess(Model):
    amplitude = attr.ib(default=1, converter=float)
    length_scale = attr.ib(default=1, converter=float)
    noise = attr.ib(default=0.1, converter=float)
    kernel = attr.ib(default=None)

    def fit(self, data):
        if self.kernel is None:
            self.kernel = C(self.amplitude, (1e-3, 1e3)) * RBF(self.length_scale, (1e-2, 1e2)) + WhiteKernel(self.noise)
        self.model = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
        points = data[self.parameters].values
        costs = data[self.experiment.__name__]
        self.model.fit(points, costs)

    def predict(self, X):
        return self.model.predict(np.atleast_2d(X), return_std = True)
