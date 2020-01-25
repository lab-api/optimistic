import inspect
from parametric import Parameter
from functools import wraps, partial
from copy import deepcopy

def search_namespace(name):
    ''' Search two frames up for the referenced Parameter. For example, if
        a parameter 'x' is defined in the local namespace and we pass a keyword
        argument x=2 to a function also defined in the local namespace, this
        function will return a handle to the parameter x.
    '''
    frame = inspect.currentframe()
    calling_namespace = frame.f_back.f_back.f_locals
    try:
        param = calling_namespace[name]
        if isinstance(param, Parameter):
            return param
    except:
        raise Exception('Parameter not found in local namespace.')


def experiment(func=None, *, ignored=[]):
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
        return partial(experiment, ignored=ignored)

    @wraps(func)
    def wrapper(*args, parallel=False, optimizer=None, **parameters):
        if not parallel:
            ## update parameters and call the decorated function
            for name, value in parameters.items():
                if name in ignored:
                    continue
                if optimizer is not None:
                    param = optimizer.parameters[name]
                else:
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
