import attr
from abc import abstractmethod
from optimistic.pipeline import Pipeline
import numpy as np

@attr.s
class Model(Pipeline):
    ''' The Model class inherits elements from both Pipeline and Block - it is a subpipeline
        which can consume data from the primary pipeline to fit a surface to observed data,
        numerically optimize the surface to inform future sampling choices, then make a physical
        measurement at the optimized point.
        '''

    @abstractmethod
    def predict(self, point):
        ''' Returns the model's prediction of the cost surface at a point X and the
            corresponding uncertainty. Should be reimplemented for each given model. '''
        return

    @abstractmethod
    def fit(self, data):
        '''Trains the model on the passed data. Reimplement for a given model. '''
        return

    def measure(self, X):
        row = dict(zip(self.parameters, np.array(X).flatten()))
        row[self.experiment.__name__] = self.predict(X)[0][0]

        self.data = self.data.append(row, ignore_index=True)
        return row[self.experiment.__name__]

    def optimize(self):
        ''' Optimizes on the response surface using the added blocks and returns
            the best point. '''
        for block in self.blocks:
            self.clone(block)
            block.measure = self.measure
            block.run()
            self.data = self.data.append(block.data)
        return self.data.iloc[self.data[self.experiment.__name__].idxmin()][self.parameters].values
