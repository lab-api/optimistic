from parametric import Parameter
from abc import abstractmethod

class Experiment:
    def __init__(self, name=''):
        self.parameters = {}
        self.name = name
        self.bounds = {}

    def set(self, **kwargs):
        for name, value in kwargs.items():
            self.parameters[name](value)

    def add_parameter(self, parameter, bounds = [None, None]):
        ''' Adds a parameter to the Experiment. If a string is passed to the "parameter"
            argument, a new Parameter is constructed with the specified value
            and bounds. If a Parameter is passed, the user can optionally pass
            a value and bounds to overwrite those on the Parameter.
        '''
        if isinstance(parameter, str):
            parameter = Parameter(parameter)
            self.set_new_bounds(parameter, bounds)
        else:
            self.set_existing_bounds(parameter, bounds)

        if parameter.name in self.__dict__:
            raise ValueError('Parameter name already exists!')

        self.parameters[parameter.name] = parameter
        setattr(self, parameter.name, parameter)

    def set_existing_bounds(self, parameter, bounds):
        ''' If no bounds were passed to add_parameter, use the Parameter's
            bounds. If these do not exist, raise a ValueError.

            If bounds were passed, check that they are within the Parameter's
            bounds.
        '''
        self.bounds[parameter.name] = [None, None]

        for i in range(2):
            if bounds[i] is None:
                if parameter.bounds[i] is not None:
                    self.bounds[parameter.name][i] = parameter.bounds[i]
                else:
                    raise ValueError(f'Assign bounds for parameter {parameter.name}.')
            else:
                if parameter.bounds[i] is None:
                    valid = True
                elif i == 0:
                    valid = bounds[i] > parameter.bounds[i]
                else:
                    valid = bounds[i] < parameter.bounds[i]
                if not valid:
                    raise ValueError(f'Bounds passed to experiment exceed global bounds for parameter "{parameter.name}".')
                self.bounds[parameter.name][i] = bounds[i]


    def set_new_bounds(self, parameter, bounds):
        ''' Assign the parameter bounds within the Experiment based on the
            passed bounds. If either limit is None, raise a ValueError.
        '''
        self.bounds[parameter.name] = [None, None]
        for i in range(2):
            if bounds[i] is None:
                raise ValueError(f'Assign bounds for parameter {parameter.name}.')
            self.bounds[parameter.name][i] = bounds[i]

    def __call__(self):
        return self.run()

    @abstractmethod
    def run(self):
        raise NotImplementedError('Define your run() method before running an experiment.')
