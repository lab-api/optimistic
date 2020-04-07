from optimistic import experiment
from parametric import Parameter, Attribute
import pytest

def test_serial_function_evaluation():
    d = Parameter('d', 2)
    @experiment
    def bar():
        return d()

    assert bar() == 2
    assert bar(d=4) == 4
    assert bar() == 4

def assert_no_parallel_function_evaluation():
    e = Parameter('e', 1)
    @experiment
    def baz():
        return e()

    with pytest.raises(IndexError):
        assert baz(parallel=True)
