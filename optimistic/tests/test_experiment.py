from optimistic import Experiment
from parametric import Parameter
import pytest

def test_add_new_parameter():
    e = Experiment()
    e.add_parameter('voltage', bounds=[0,1])
    assert 'voltage' in e.parameters
    assert e.bounds['voltage'] == [0,1]

def test_add_existing_parameter():
    e = Experiment()
    phase = Parameter('phase', bounds=[0, 6.28])
    e.add_parameter(phase)
    assert 'phase' in e.parameters
    assert e.bounds['phase'] == [0,6.28]

def test_add_duplicate_parameter():
    e = Experiment()
    e.add_parameter('voltage', bounds=[0,1])
    with pytest.raises(ValueError):
        assert e.add_parameter('voltage', bounds=[0,1])
