import numpy as np
import pandas as pd
import attr

@attr.s
class Algorithm:
    experiment = attr.ib()
    parameters = attr.ib(factory=dict)
    bounds = attr.ib(factory=dict)
    data = attr.ib(factory=pd.DataFrame)

    def add_parameter(self, parameter, bounds=None):
        self.parameters[parameter.name] = parameter
        if bounds is None:
            if parameter.bounds == (-np.inf, np.inf):
                raise ValueError('Define parameter bounds!')
            self.bounds[parameter.name] = parameter.bounds
        else:
            self.bounds[parameter.name] = bounds

        return self

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
            i += 1

    def measure(self, point):
        ''' Actuate to specified point and measure result '''
        self.actuate(point)
        i = len(self.data)
        for name, parameter in self.parameters.items():
            self.data.loc[i, name] = parameter()
        self.data.loc[i, 'result'] = self.experiment()
        return self.data.loc[i, 'result']

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
