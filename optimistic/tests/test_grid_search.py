from parametric import Parameter
from optimistic.algorithms import GridSearch
from optimistic import experiment
import numpy as np

def test_grid_search():
    x = Parameter('x')

    @experiment
    def result():
        return x**2

    opt = GridSearch(result, steps=5)
    opt.add_parameter(x, bounds=(-1, 1))
    opt.run()
    assert (opt.X.flatten() == [-1, -0.5, 0, 0.5, 1]).all()
