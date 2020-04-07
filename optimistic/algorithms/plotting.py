import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, algorithm):
        self.data = algorithm.dataset
        self.experiment = algorithm.experiment

    def convergence(self):
        self.data[self.experiment.__name__].plot()
        plt.xlabel('Iteration')
        plt.ylabel(self.experiment.__name__)

    def parameter_space(self, parameter):
        ''' Pass up to two parameters to visualize the parameter space of the objective function '''
        plt.plot(self.data[parameter.name], self.data[self.experiment.__name__], '.')
        plt.xlabel(parameter.name)
        plt.ylabel(self.experiment.__name__)

    def curves(self, x, y):
        z = self.experiment.__name__
        for x0, df in self.data.groupby(y):
            plt.plot(df[x], df[z], label=f'{y}={np.round(x0, 3)}')
        plt.legend(loc=(1.04,0))
        plt.xlabel(x)
        plt.ylabel(z)
