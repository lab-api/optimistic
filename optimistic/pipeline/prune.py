import attr
from optimistic.algorithms import Algorithm

@attr.s
class Prune(Algorithm):
    ''' The Prune block removes points in the Pipeline.data set which are
        above a threshold cost (as a fraction of the best found cost). This
        can be used to discard low-impact points (far from known minima)
        to improve model training efficiency.
    '''
    threshold = attr.ib(default=0.5, converter=float)

    def run(self):
        costs = self.parent.data[self.experiment.__name__]
        threshold = self.threshold * costs.min()
        self.parent.data = self.parent.data[costs < threshold].reset_index()
