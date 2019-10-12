import attr
from optimistic.algorithms import Algorithm

@attr.s
class Zoom(Algorithm):
    ''' The Zoom block restricts the bounds of each parameter only to regions
        above a threshold cost. This can be used, for example, in successive
        grid searches with decreasing search range.
    '''
    threshold = attr.ib(default=0.5, converter=float)

    def run(self):
        costs = self.parent.data[self.experiment.__name__]
        threshold = costs.min() * self.threshold
        for p in self.parameters:
            valid_points = self.parent.data[costs < threshold][p]
            self.parent.bounds[p] = (valid_points.min(), valid_points.max())
