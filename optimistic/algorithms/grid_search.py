from optimistic.algorithms import Algorithm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import attr

@attr.s
class GridSearch(Algorithm):
    steps = attr.ib(default=20, converter=int)
    scans = attr.ib(default=1, converter=int)

    def run(self):
        dim = len(self.parameters)
        grid = []

        for name, parameter in self.parameters.items():
            if name in self.points:
                scaler = MinMaxScaler()
                bounds = np.array(self.bounds[name])
                points = np.array(self.points[name])
                scaler.fit(bounds.reshape(-1, 1))
                normalized_points = scaler.transform(points.reshape(-1, 1)).flatten()
                grid.append(normalized_points)
            else:
                grid.append(np.linspace(0, 1, self.steps))

        points = np.transpose(np.meshgrid(*[grid[n] for n in range(dim)])).reshape(-1, dim)

        for point in points:
            self.measure(point)
