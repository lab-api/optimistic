from optimistic import experiment
from parametric import Parameter, Attribute, parametrize
import pytest

def test_method_evaluation():
    @parametrize
    class Foo:
        a = Attribute('a', 1)
        b = Attribute('b', 2)

        @experiment
        def bar(self):
            return self.a()

    foo = Foo()

    assert foo.bar(a=2) == 2
    assert foo.bar(a=3, parallel=True) == 3   # evaluate in new instance
    assert foo.bar() == 2      # check that parallel evaluation left a unchanged

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
