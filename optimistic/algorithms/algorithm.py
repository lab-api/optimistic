import numpy as np
import pandas as pd
import attr
import time
from tqdm.auto import tqdm
from optimistic import experiment as objective
from .plotting import Plotter
from threading import Thread
from ipywidgets import Text

@attr.s
class Algorithm:
    ''' A base class for parameter space exploration and optimization.

        Arguments:
            experiment (callable): a function or method which measures the
                                   objective function at the current point in
                                   the parameter space. Takes no positional arguments.
            parameters (dict): instances of the parametric.Parameter class to use
                               during optimization. Defaults to empty, and Parameters
                               can be added by calling Algorithm.add_parameter().
            bounds (dict): tuple bounds indexed by Parameter names. Defaults to empty,
                           and is set when adding Parameters.
            points (dict): optional points for each Parameter to override default
                           point generation in optimizers, e.g. sampling locations
                           for a grid search or initial population in a genetic algorithm.
            sign (int): choose between maximization (+1) and minimization (-1)
            X (2d array): coordinates sampled during optimization. Defaults to empty,
                          but previous results can be passed (along with results into
                          the y argument) to speed up some optimizers.
            y (1d array): objective function evaluations. Defaults to empty.
            threaded (bool): if True, run optimization in a separate thread.
            show_progress (bool): whether to display a progress bar during
                                  optimization. Adds <1 ms overhead per iteration.
            record_data (bool): whether to store X, y observations. Adds <1 ms overhead per iteration.
            display (bool): whether to display optimization status using ipywidgets.
                            Only works when running in a Jupyter environment.
            continuous (bool): whether to quit after convergence/specified number of iterations
                               or continue running.
    '''
    experiment = attr.ib(default=None)
    parameters = attr.ib(factory=dict)
    bounds = attr.ib(factory=dict)
    points = attr.ib(factory=dict)        # optional overrides to search points
    sign = attr.ib(default=1, converter=np.sign)
    X = attr.ib(factory=lambda: np.atleast_2d([]))
    y = attr.ib(factory=lambda: np.array([]))

    threaded = attr.ib(default=False)
    show_progress = attr.ib(default=True)
    record_data = attr.ib(default=True)
    display = attr.ib(default=False)
    continuous = attr.ib(default=False)

    output = attr.ib(default=None)

    def add_parameter(self, parameter, bounds=None, points=None):
        ''' Adds a parameter.

            Arguments:
                parameter (parametric.Parameter)
                bounds (tuple): a (min, max) pair defining the extent of the
                                optimization. If no bounds are passed but a set of
                                points is specified, use the min/max values of
                                points; otherwise, use the default parameter bounds.
                points (array-like): a list of points to override sampling behavior
                                     in the algorithm.
        '''
        self.parameters[parameter.name] = parameter
        if bounds is None:
            if points is not None:
                self.bounds[parameter.name] = (np.min(points), np.max(points))
            elif parameter.bounds == (-np.inf, np.inf):
                raise ValueError('Define parameter bounds!')
            else:
                self.bounds[parameter.name] = parameter.bounds
        else:
            self.bounds[parameter.name] = bounds

        if points is not None:       # override point selection
            self.points[parameter.name] = points

        return self

    def check_bounds(self, point):
        ''' Checks that the point is within the specified bounds '''
        i = 0
        for name, parameter in self.parameters.items():
            bounds = self.bounds[name]
            if not bounds[0] <= point[i] <= bounds[1]:
                raise ValueError(f'The optimizer requested a point outside the valid bounds for parameter {parameter.name} and will now terminate.')
            i += 1

    def measure(self, point):
        ''' Actuate to specified point and measure result '''
        if self.experiment is None:
            raise ValueError('No experiment has been assigned to this optimizer!')

        self.check_bounds(point)

        new_values = {}
        for i, (name, parameter) in enumerate(self.parameters.items()):
            new_values[name] = point[i]
        result = self.experiment(optimizer=self, **new_values)

        if self.record_data:
            if len(self.X[0]) == 0:
                self.X = np.atleast_2d(point)
            else:
                self.X = np.append(self.X, np.atleast_2d(point), axis=0)
            self.y = np.append(self.y, result)

        if self.display:
            if self.output is None:
                self.output = Text()
                display(self.output)
            self.output.value = str(point) + ' -> ' + str(result)

        return -self.sign*result

    @property
    def dataset(self):
        ''' If the optimizer is set to data_format = 'numpy', this converts acquired
            data into a pandas.DataFrame.
        '''
        df = pd.DataFrame(self.X, columns = list(self.parameters.keys()))
        df[self.experiment.__name__] = self.y
        return df

    @objective
    def metacost(self):
        ''' Runs an optimization and returns the integrated cost as an objective
            function for meta-optimization. The @objective tag supports passing
            any of the optimizer parameters into the metacost evaluation.
        '''
        # measure original coordinates
        original_coordinates = {}
        for p in self.parameters.values():
            original_coordinates[p.name] = p()

        # run experiment and compute integral of objective function
        self.data = pd.DataFrame()
        self.run()
        integrated_cost = self.data[self.experiment.__name__].sum()

        # reset coordinates
        for p in original_coordinates:
            self.parameters[p](original_coordinates[p])

        return -integrated_cost

    @property
    def plot(self):
        return Plotter(self)

    @classmethod
    def study(cls, experiment, parameter, bounds, **kwargs):
        ''' Launches a 1D optimization of the passed experiment with
            the given parameter and bounds. You can include any keyword arguments
            you want to pass to the optimizer.
        '''
        inst = cls(experiment, **kwargs)
        inst.add_parameter(parameter, bounds=bounds)
        inst.run()
        inst.plot.parameter_space(parameter)
        return inst

    def range(self, iterations):
        ''' Functions similarly to the built-in range() generator, e.g.
                for i in self.range(10):
                    print(i)
            prints integers from 0 to 9. However, if self.continuous==True,
            i will be reset to 0 afterwards and the integers will be repeatedly
            printed in order until execution is interrupted.
        '''
        if not self.continuous:
            yield from range(1*iterations)
        else:
            i = 0
            while True:
                yield i
                i = (i+1) % 1*iterations

    def iterate(self, lst):
        ''' Functions similarly to the built-in list() generator, e.g.
                for x in [1, 2, 3]:
                    print(x)
            prints 1, 2, and 3. However, if self.continuous==True,
            this function will continue to loop through the list values.

            If self.show_progress==True, returns a tqdm generator for displaying
            a progress bar.
        '''
        if self.show_progress and not self.continuous:
            yield from tqdm(lst)
        elif not self.continuous:
            yield from list(lst)
        else:
            i = 0
            while True:
                yield lst[i]
                i = (i+1) % len(lst)

    def run(self):
        if self.threaded:
            Thread(target=self._run).start()
        else:
            self._run()
