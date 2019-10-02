from optimistic.algorithms import Algorithm
import numpy as np
import attr

@attr.s
class GradientDescent(Algorithm):
    iterations = attr.ib(default=100, converter=int)
    learning_rate = attr.ib(default=1e-3, converter=float)
    dither_size = attr.ib(1e-2, converter=float)

    def gradient(self, x):
        dim = len(self.parameters)
        g = np.zeros(dim)
        for d in range(dim):
            step = np.zeros(dim)
            step[d] = self.dither_size

            c1 = self.measure(x+step)
            c2 = self.measure(x-step)
            g[d] = (c1-c2)/(2*self.dither_size)

        return g

    def run(self):
        x_i = self.normalize([p() for p in self.parameters.values()])

        for i in range(self.iterations):
            x_i -= self.learning_rate * self.gradient(x_i)
            self.measure(x_i)
