import numpy as np
import pandas as pd

class Algorithm:
    def __init__(self, experiment, parameters):
        ''' Base algorithm class.
            Args:
                experiment (Experiment): an instance of the Experiment class
                parameters (list): handles of the Parameter instances to be varied
        '''
        self.experiment = experiment
        self.parameters = self.prepare_parameters(parameters)
        cols = [x for x in self.parameters]
        cols.append('result')
        self.data = pd.DataFrame(columns=cols)

    def prepare_parameters(self, parameters):
        ''' Registers the passed parameters with the optimizer. Parameters can
            either be passed as instances of the Parameter class or as strings
            referring to attributes of the experiment.
        '''
        parameters_dict = {}
        for p in parameters:
            if isinstance(p, str):
                try:
                    p = self.experiment.parameters[p]
                except KeyError:
                    raise KeyError('Parameter "{p}" not defined within the experiment "{self.experiment.__name__}".')
            parameters_dict[p.name] = p
        return parameters_dict

    def actuate(self, point):
        ''' Sets each parameter to the value corresponding to a normalized value
            in the passed array.
        '''
        point = self.unnormalize(point)

        i = 0
        for name, parameter in self.parameters.items():
            bounds = self.experiment.bounds[name]
            if point[i] < bounds[0] or point[i] > bounds[1]:
                raise ValueError(f'The optimizer requested a point outside the valid bounds for parameter {p.name} and will now terminate.')
            parameter(point[i])
            i += 1

    def measure(self, point):
        ''' Actuate to specified point and measure result '''
        self.actuate(point)
        i = len(self.data)
        for p in self.parameters.values():
            self.data.loc[i, p.name] = p()
        self.data.loc[i, 'result'] = self.experiment()
        return self.data.loc[i, 'result']

    def normalize(self, points):
        ''' Normalize a point to (0,1) according to the optimizer bounds.
            Return the point in the same format it was passed.
        '''
        shape = np.shape(points)
        a = np.empty(np.atleast_2d(points).shape)
        if a.shape[1] != len(self.parameters):
            raise Exception('Dimension of array passed to unnormalize() is incompatible with number of Parameters.')
        i=0
        for p in self.parameters.values():
            bounds = self.experiment.bounds[p.name]
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
            bounds = self.experiment.bounds[p.name]
            a[:,i] = bounds[0] + np.atleast_2d(points)[:,i]*(bounds[1]-bounds[0])
            i += 1

        if len(shape) == 0:     # single value was passed
            return a[0,0]
        elif len(shape) == 1:  # 1D list/array was passed
            return a[0]
        else:
            return a            # 2D list/array was passed
