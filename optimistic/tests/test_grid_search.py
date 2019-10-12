from parametric import Parameter
from optimistic.algorithms import GridSearch
import numpy as np

x = Parameter('x')
def result():
    return x**2

def test_grid_search():
    opt = GridSearch(result, steps=5)
    opt.add_parameter(x, bounds=(-1, 1))
    opt.run()
    assert (opt.data['x'] == [-1, -0.5, 0, 0.5, 1, 0]).all()
    assert opt.experiment() == opt.data['result'].min()
