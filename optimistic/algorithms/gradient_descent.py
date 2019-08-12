from optimistic.algorithms import Algorithm
import numpy as np

class GradientDescent(Algorithm):
    def __init__(self, experiment, parameters, iterations=100, learning_rate=1e-3, dither_size=1e-2):
        super().__init__(experiment, parameters)
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.dither_size = float(dither_size)

        self.run()

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
