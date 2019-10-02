from optimistic.algorithms import Algorithm
from parametric import Parameter
import pytest

def result():
    return 0


def test_add_existing_parameter():
    e = Algorithm(result)
    phase = Parameter('phase', bounds=[0, 6.28])
    e.add_parameter(phase)
    assert 'phase' in e.parameters
    assert e.bounds['phase'] == [0,6.28]
