import inspect
from parametric import Parameter
from functools import wraps, partial
from copy import deepcopy

def search_namespace(name, frame=None):
    ''' Recursively search upwards through namespaces for the referenced Parameter '''
    if frame is None:
        frame = inspect.currentframe()
    calling_namespace = frame.f_back.f_locals
    if name not in calling_namespace:
        return search_namespace(name, frame.f_back)
    else:
        param = calling_namespace[name]
        if isinstance(param, Parameter):
            return param
        else:
            return search_namespace(name, frame.f_back)

def experiment(func=None, *, ignored=[], parallel=False):
    ''' Decorates an experiment function, which by default should have no positional
        or keyword arguments. Adds optional keyword arguments which are forwarded
        to update parameter values, which are searched by name recursively
        upwards from the calling frame.

        Passing a list of parameter names to the "ignored" argument overrides
        the default pre-experiment actuation, allowing custom actuation to be
        written in the experiment function.

        For parallel simulations, the experiment should be a method of a class
        containerizing the parameters. Passing parallel=True runs the decorated
        function in a new instance of the class, allowing simultaneous evaluation
        of different parameter sets.
    '''
    if func is None:
        return partial(experiment, ignored=ignored, parallel=parallel)

    @wraps(func)
    def wrapper(*args, **parameters):
        if not parallel:
            ## update parameters and call the decorated function
            for name, value in parameters.items():
                if name in ignored:
                    continue
                ## if name corresponds to a parameter of the parent class, update that
                try:
                    param = getattr(args[0], name)
                    assert isinstance(param, Parameter)
                except:
                    ## otherwise, search namespace
                    param = search_namespace(name)
                param(value)
            return func(*args)
        else:
            ## clone the instance, then update parameters and call the cloned function
            if len(args) == 0:
                raise IndexError('Parallel optimization is supported for class methods only to ensure Parameter containerization.')
            clone = deepcopy(args[0])
            for name, value in parameters.items():
                if name in ignored:
                    continue
                param = getattr(clone, name)
                param(value)
            return func(clone)
    return wrapper
