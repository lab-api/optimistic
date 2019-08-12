from parametric import Parameter
from optimistic import Experiment
from optimistic.algorithms import Algorithm
import pytest
import numpy as np

def test_invalid_parameter():
    ''' Check that a KeyError is raised if an invalid parameter is passed to the Algorithm. '''
    e = Experiment()
    with pytest.raises(KeyError):
        assert Algorithm(e, parameters=['parameter'])

def test_normalization():
    e = Experiment()
    voltage_bounds = [0.25, 0.75]
    current_bounds = [0.5, 1]
    e.add_parameter('voltage', bounds=voltage_bounds)
    e.add_parameter('current', bounds=current_bounds)
    opt = Algorithm(e, ['voltage', 'current'])

    assert (opt.normalize([voltage_bounds[0],current_bounds[0]]) == [0, 0]).all()
    assert (opt.normalize([voltage_bounds[0],current_bounds[1]]) == [0, 1]).all()
    assert (opt.normalize([voltage_bounds[1],current_bounds[0]]) == [1, 0]).all()
    assert (opt.normalize([voltage_bounds[1],current_bounds[1]]) == [1, 1]).all()

    with pytest.raises(Exception):
        assert opt.normalize([1,2,3])

def test_unnormalization():
    e = Experiment()
    voltage_bounds = [0.25, 0.75]
    current_bounds = [0.5, 1]
    e.add_parameter('voltage', bounds=voltage_bounds)
    e.add_parameter('current', bounds=current_bounds)
    opt = Algorithm(e, ['voltage', 'current'])

    assert (opt.unnormalize([0,0]) == [voltage_bounds[0], current_bounds[0]]).all()
    assert (opt.unnormalize([0,1]) == [voltage_bounds[0], current_bounds[1]]).all()
    assert (opt.unnormalize([1,0]) == [voltage_bounds[1], current_bounds[0]]).all()
    assert (opt.unnormalize([1,1]) == [voltage_bounds[1], current_bounds[1]]).all()

    with pytest.raises(Exception):
        assert opt.unnormalize([1,2,3])

def test_actuation():
    e = Experiment()
    e.add_parameter('voltage', bounds=[0,1])
    e.add_parameter('current', bounds=[0,1])
    opt = Algorithm(e, ['voltage', 'current'])

    new_voltage = np.random.uniform()
    new_current = np.random.uniform()
    opt.actuate([new_voltage, new_current])
    assert opt.parameters['voltage']() == new_voltage
    assert opt.parameters['current']() == new_current
