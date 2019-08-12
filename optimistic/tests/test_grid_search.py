from optimistic import Experiment
from optimistic.algorithms import GridSearch
import numpy as np

class QuadraticExperiment(Experiment):
    def __init__(self):
        super().__init__()
        self.add_parameter('x', bounds=[-1,1])

    def __call__(self):
        return self.parameters['x']()**2

def test_grid_search():
    e = QuadraticExperiment()
    opt = GridSearch(e, ['x'], steps=5)

    assert (opt.data['x'] == [-1, -0.5, 0, 0.5, 1]).all()
    assert opt.experiment() == opt.data['result'].min()
