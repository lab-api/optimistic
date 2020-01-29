from optimistic.algorithms import Algorithm
import numpy as np
import attr
import pandas as pd
import dask
import multiprocessing as mp
from dask.distributed import Client

@attr.s
class GridSearch(Algorithm):
    steps = attr.ib(default=20, converter=int)
    scans = attr.ib(default=1, converter=int)
    parallel = attr.ib(default=False, converter=bool)
    threads_per_worker = attr.ib(default=1, converter=int)
    workers = attr.ib(default=mp.cpu_count(), converter=int)
    logarithmic = attr.ib(default=False, converter=bool)

    def generate_grid(self):
        dim = len(self.parameters)
        grid = []
        for name, parameter in self.parameters.items():
            if name in self.points:
                grid.append(self.points[name])
            else:
                if self.logarithmic:
                    grid.append(np.logspace(np.log10(self.bounds[name][0]),
                                            np.log10(self.bounds[name][1]),
                                            self.steps))
                else:
                    grid.append(np.linspace(self.bounds[name][0],
                                            self.bounds[name][1],
                                            self.steps))
        return np.transpose(np.meshgrid(*[grid[n] for n in range(dim)])).reshape(-1, dim)

    def _run(self):
        if self.parallel:
            self.run_parallel()
        else:
            self.run_sequential()

    def run_sequential(self):
        points = self.generate_grid()
        for point in self.iterate(points):
            self.measure(point)

    def run_parallel(self):
        points = self.generate_grid()
        points = pd.DataFrame(points, columns = list(self.parameters.keys()))
        client = Client(threads_per_worker=self.threads_per_worker, n_workers=self.workers)
        display(client)
        futures = []
        for values in points.values:
            future = client.submit(self.experiment, parallel=True, **dict(zip(self.parameters, values)))
            futures.append(future)
        results = client.gather(futures)

        points[self.experiment.__name__] = results
        self.data = points
