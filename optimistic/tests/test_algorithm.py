from parametric import Parameter
from optimistic.algorithms import Algorithm
from optimistic import experiment
import pytest
import numpy as np

@experiment
def result():
    return 0

opt = Algorithm(result)
voltage = Parameter('voltage')
current = Parameter('current')
voltage_bounds = [0.25, 0.75]
current_bounds = [0.5, 1]
opt.add_parameter(voltage, bounds=voltage_bounds)
opt.add_parameter(current, bounds=current_bounds)

def test_normalization():
    assert (opt.normalize([voltage_bounds[0],current_bounds[0]]) == [0, 0]).all()
    assert (opt.normalize([voltage_bounds[0],current_bounds[1]]) == [0, 1]).all()
    assert (opt.normalize([voltage_bounds[1],current_bounds[0]]) == [1, 0]).all()
    assert (opt.normalize([voltage_bounds[1],current_bounds[1]]) == [1, 1]).all()

    with pytest.raises(Exception):
        assert opt.normalize([1,2,3])


def test_unnormalization():
    assert (opt.unnormalize([0,0]) == [voltage_bounds[0], current_bounds[0]]).all()
    assert (opt.unnormalize([0,1]) == [voltage_bounds[0], current_bounds[1]]).all()
    assert (opt.unnormalize([1,0]) == [voltage_bounds[1], current_bounds[0]]).all()
    assert (opt.unnormalize([1,1]) == [voltage_bounds[1], current_bounds[1]]).all()

    with pytest.raises(Exception):
        assert opt.unnormalize([1,2,3])
