import numpy as np
import pandas as pd
import attr
import time

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
    experiment = attr.ib()
    parameters = attr.ib(factory=dict)
    bounds = attr.ib(factory=dict)
    points = attr.ib(factory=dict)        # optional overrides to search points
    delays = attr.ib(factory=dict)        # post-actuation delays
    data = attr.ib(factory=pd.DataFrame)
    dependent_variables = attr.ib(factory=dict)

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
        point = self.unnormalize(point)

        i = 0
        for name, parameter in self.parameters.items():
            bounds = self.bounds[name]
            if not bounds[0] <= point[i] <= bounds[1]:
                raise ValueError(f'The optimizer requested a point outside the valid bounds for parameter {parameter.name} and will now terminate.')
            parameter(point[i])
            time.sleep(self.delays[name])
            i += 1

    def result_to_dataframe(self, result):
        ''' Takes an experimental result and forms a DataFrame to append to self.data. '''
        ## case 1: type(result) == float
        if isinstance(result, float):
            new_data = pd.DataFrame(index=[len(self.data)], columns=[*list(self.parameters), self.experiment.__name__])
            for name, parameter in self.parameters.items():
                new_data[name] = parameter()
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


    def measure(self, point):
        ''' Actuate to specified point and measure result '''
        self.actuate(point)
        new_data = self.result_to_dataframe(self.experiment())
        self.data = self.data.append(new_data)
        self.data = self.data.reset_index().drop('index', axis=1)

        return self.data.iloc[-1][self.experiment.__name__]

    def normalize(self, points):
        ''' Normalize a point to (0,1) according to the optimizer bounds.
            Return the point in the same format it was passed.
        '''
        shape = np.shape(points)
        a = np.empty(np.atleast_2d(points).shape)
        if a.shape[1] != len(self.parameters):
            raise Exception('Dimension of array passed to normalize() is incompatible with number of Parameters.')
        i=0
        for p in self.parameters.values():
            bounds = self.bounds[p.name]
            a[:,i] = (np.atleast_2d(points)[:,i] - bounds[0]) / (bounds[1]-bounds[0])
            i += 1

        if len(shape) == 0:     # single value was passed
            return a[0,0]
        elif len(shape) == 1:  # 1D list/array was passed
            return a[0]
        else:                   # 2D list/array was passed
            return a

    def unnormalize(self, points):
        ''' Convert a normalized (0-1) point according to the optimizer bounds.
            Return the point in the same format it was passed.
        '''
        shape = np.shape(points)
        a = np.empty(np.atleast_2d(points).shape)
        if a.shape[1] != len(self.parameters):
            raise Exception('Dimension of array passed to unnormalize() is incompatible with number of Parameters.')
        i=0
        for p in self.parameters.values():
            bounds = self.bounds[p.name]
            a[:,i] = bounds[0] + np.atleast_2d(points)[:,i]*(bounds[1]-bounds[0])
            i += 1

        if len(shape) == 0:     # single value was passed
            return a[0,0]
        elif len(shape) == 1:  # 1D list/array was passed
            return a[0]
        else:
            return a            # 2D list/array was passed
