import attr
import inspect
import numpy as np
from scipy.optimize import curve_fit
from optimistic.models import Model

@attr.s
class Surface(Model):
    ''' An arbitrary user-specified response surface. '''
    surface = attr.ib(default=None)
    p0 = attr.ib(default=None)

    def fit(self, data):
        self.data = data
        if self.surface is None:
            raise ValueError('Specify a surface!')
        dof = len(inspect.signature(self.surface).parameters)-1
        if self.p0 is None:
            self.p0 = tuple([0.5]*dof)
        points = data[self.parameters].values
        costs = data[self.experiment.__name__].values
        self.popt, self.pcov = curve_fit(self.objective_function, points, costs, self.p0)

    def objective_function(self, X, *args):
        return self.surface(np.atleast_2d(X), *args)

    def predict(self, X):
        return self.surface(X, *tuple(self.popt)), 0
