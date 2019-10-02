from optimistic.algorithms import Algorithm
import numpy as np
import attr

@attr.s
class GridSearch(Algorithm):
    steps = attr.ib(default=20, converter=int)
    scans = attr.ib(default=1, converter=int)

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
