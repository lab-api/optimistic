from optimistic.algorithms import Algorithm
import numpy as np
import attr
import pandas as pd
import dask
import multiprocessing as mp
from dask.distributed import Client

@attr.s
class ParallelGridSearch(Algorithm):
    ''' Run a grid search across a set of parameters in parallel. Since separate simulation
        instances are constructed for each parameter set, the Algorithm.experiment object
        should have keyword arguments corresponding to each parameter in the simulation. Note
        that this is different from regular optimizers, which take experiments with no parameters.

        This difference could be unified by adding a decorator to normal argument-less parameters
    '''
    steps = attr.ib(default=20, converter=int)
    scans = attr.ib(default=1, converter=int)
    threads_per_worker = attr.ib(default=1, converter=int)
    workers = attr.ib(default=mp.cpu_count(), converter=int)

    def run(self):
        dim = len(self.parameters)
        grid = []

        for name, parameter in self.parameters.items():
            if name in self.points:
                grid.append(self.points[name])
            else:
                grid.append(np.linspace(self.bounds[name][0], self.bounds[name][1], self.steps))

        points = np.transpose(np.meshgrid(*[grid[n] for n in range(dim)])).reshape(-1, dim)

        points = pd.DataFrame(points, columns = list(self.parameters.keys()))

        client = Client(threads_per_worker=self.threads_per_worker, n_workers=self.workers)
        display(client)
        futures = []
        for values in points.values:
            future = client.submit(self.experiment, **dict(zip(self.parameters, values)))
            futures.append(future)
        results = client.gather(futures)

        points[self.experiment.__name__] = results
        self.data = points
