import numpy as np
import pandas as pd
import attr
import time
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler
from optimistic import experiment as objective
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

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
            delays (dict): post-actuation delay. Set in add_parameter and defaults to zero.
                           Should be set in Algorithm.add_parameter() if a delay is
                           desired between actuation and measurement.
            data (pandas.DataFrame): results of the optimization. Defaults to empty,
                                     but a previously-generated DataFrame can be
                                     passed to speed up optimization of a stationary
                                     objective function.
            dependent_variables (dict): a labeled set of functions to measure
                                        along-side the objective function. For
                                        example, it may be desirable to monitor
                                        laser power while optimizing the frequency.
    '''
    experiment = attr.ib(default=None)
    parameters = attr.ib(factory=dict)
    bounds = attr.ib(factory=dict)
    points = attr.ib(factory=dict)        # optional overrides to search points
    delays = attr.ib(factory=dict)        # post-actuation delays
    data = attr.ib(factory=pd.DataFrame)
    dependent_variables = attr.ib(factory=dict)
    show_progress = attr.ib(default=True)

    @property
    def data_normalized(self):
        ''' Return a numpy array corresponding to the parameter space points in
            self.data. '''
        normalized_data = self.data.copy()
        normalized_array = self.normalize(self.data[self.parameters].values)
        for i, x in enumerate(self.parameters):
            normalized_data[x] = normalized_array[:, i]

        return normalized_data

    def add_parameter(self, parameter, bounds=None, points=None, delay=0):
        ''' Adds a parameter.

            Arguments:
                parameter (parametric.Parameter)
                bounds (tuple): a (min, max) pair defining the extent of the
                                optimization. If no bounds are passed but a set of
                                points is specified, use the min/max values of
                                points; otherwise, use the default parameter bounds.
                points (array-like): a list of points to override sampling behavior
                                     in the algorithm.
                delay (float): optional delay after actuation.
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

        self.delays[parameter.name] = delay
        return self

    def add_dependent_variable(self, function):
        self.dependent_variables.append(function)

    def actuate(self, point):
        ''' Sets each parameter to the value corresponding to a normalized value
            in the passed array.
        '''
        point = self.unnormalize(point)[0]

        i = 0
        for name, parameter in self.parameters.items():
            bounds = self.bounds[name]
            if not np.round(bounds[0], 7) <= np.round(point[i], 7) <= np.round(bounds[1], 7):
                print(point[i], bounds[0], bounds[0] <= point[i])
                print(point[i], bounds[1], point[i] <= bounds[1])
                raise ValueError(f'The optimizer requested a point outside the valid bounds for parameter {parameter.name} and will now terminate.')
            parameter(point[i])
            time.sleep(self.delays[name])
            i += 1

    def result_to_dataframe(self, dependent_variables, result):
        ''' Takes an experimental result and forms a DataFrame to append to self.data. '''
        ## case 1: type(result) == float
        if isinstance(result, float) or isinstance(result, int):
            new_data = pd.DataFrame(index=[len(self.data)], columns=[*list(self.parameters), self.experiment.__name__])
            for name, parameter in self.parameters.items():
                new_data[name] = dependent_variables[name]
            new_data[self.experiment.__name__] = result

        if isinstance(result, pd.DataFrame):
            ''' It's expected that self.experiment return a DataFrame with columns
                corresponding to self.experiment.__name__ and zero or more
                other variables to be tracked (not contained in self.dependent_variables).
            '''
            new_data = result
            for name, parameter in self.parameters.items():
                new_data[name] = parameter()

        ## measure dependent variables
        for function in self.dependent_variables:
            new_data[function.__name__] = function()

        return new_data

    def measure(self, point, normalized=True):
        ''' Actuate to specified point and measure result '''
        if self.experiment is None:
            raise ValueError('No experiment has been assigned to this optimizer!')

        new_values = {}
        if normalized:
            point = self.unnormalize(point)[0]
        for i, (name, parameter) in enumerate(self.parameters.items()):
            new_values[name] = point[i]
        new_data = self.result_to_dataframe(new_values, self.experiment(optimizer=self, **new_values))

        self.data = self.data.append(new_data)
        self.data = self.data.reset_index().drop('index', axis=1)

        return self.data.iloc[-1][self.experiment.__name__]

    def scaler(self):
        bounds_array = np.atleast_2d([x for x in self.bounds.values()])
        scaler = MinMaxScaler()
        scaler.fit(bounds_array.astype(float).T)

        return scaler

    def normalize(self, points):
        ''' Normalize a point to (0,1) according to the optimizer bounds.
            Return a 2D array.
        '''
        return self.scaler().transform(np.atleast_2d(points).astype(float))

    def unnormalize(self, points):
        ''' Unnormalize a point from (0,1) according to the optimizer bounds.
            Return a 2D array.
        '''
        return self.scaler().inverse_transform(np.atleast_2d(points).astype(float))

    @property
    def plot(self):
        if plt is None:
            raise ImportError('Matplotlib could not be imported.')
        from .plotting import Plotter
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

    def iterate(self, lst):
        if self.show_progress:
            return tqdm(lst)
        else:
            return lst
