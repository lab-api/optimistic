import numpy as np
import attr
from parametric import Attribute
from optimistic.algorithms import Algorithm

@attr.s
class GradientDescent(Algorithm):
    iterations = Attribute('iterations', 100, converter=int)
    learning_rate = Attribute('learning_rate', 1e-3, converter=float)
    dither_size = Attribute('dither_size', 1e-2, converter=float)

    def gradient(self, x):
        dim = len(self.parameters)
        g = np.zeros(dim)
        for d in range(dim):
            step = np.zeros(dim)
            step[d] = self.dither_size()

            c1 = self.measure(x+step)
            c2 = self.measure(x-step)
            g[d] = (c1-c2)/(2*self.dither_size)

        return g

    def run(self):
        x_i = self.normalize([p() for p in self.parameters.values()])

        for i in range(self.iterations()):
            x_i -= self.learning_rate * self.gradient(x_i)
            self.measure(x_i)
