from optimistic.algorithms import Algorithm
import numpy as np

class GridSearch(Algorithm):
    def __init__(self, experiment, parameters, steps=20, scans=1):
        super().__init__(experiment, parameters)
        self.steps = steps
        
        self.run()

    def run(self):
        dim = len(self.parameters)
        grid = []
        for n in range(dim):
            grid.append(np.linspace(0, 1, self.steps))
        points = np.transpose(np.meshgrid(*[grid[n] for n in range(dim)])).reshape(-1, dim)

        for point in points:
            self.measure(point)

        best_point = points[self.data['result'].idxmin()]
        self.actuate(best_point)
