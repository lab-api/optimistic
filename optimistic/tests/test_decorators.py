from optimistic.decorators import factory, experiment
from parametric import Parameter
import pytest

@factory
class Foo:
    a = Parameter('a', 1)
    b = Parameter('b', 2)

    @experiment(parallel=True)
    def bar(self):
        return self.a()

    @experiment
    def baz(self):
        return self.b()

foo = Foo()

def test_parallel_method_evaluation():
    assert foo.bar() == 1
    assert foo.bar(a=3) == 3   # evaluate in new instance
    assert foo.bar() == 1      # check that parallel evaluation left a unchanged

def test_serial_method_evaluation():
    assert foo.baz() == 2
    assert foo.baz(b=4) == 4   # evaluate with updated b
    assert foo.baz() == 4      # check that the update persists

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
    @experiment(parallel=True)
    def baz():
        return e()

    with pytest.raises(IndexError):
        assert baz()
