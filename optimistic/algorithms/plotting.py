import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, algorithm):
        self.data = algorithm.data
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
