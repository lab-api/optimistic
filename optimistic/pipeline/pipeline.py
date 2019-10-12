''' The Pipeline module allows custom optimization sequences to be strung together
    from basic blocks. For example, a coarse grid search could be employed to
    find an initial signal, followed by a gradient descent. '''

import attr
from optimistic.algorithms import Algorithm

@attr.s
class Pipeline(Algorithm):
    blocks = attr.ib(factory=list)

    def add_block(self, block):
        self.blocks.append(block)

    def clone(self, block):
        ''' Copies over settings from the Pipeline to the block. '''
        block.experiment = self.experiment
        block.parameters = self.parameters
        block.bounds = self.bounds
        block.points = self.points
        block.delays = self.delays
        block.dependent_variables = self.dependent_variables

    def run(self):
        from optimistic.models import Model

        for block in self.blocks:
            self.clone(block)
            if isinstance(block, Model):
                block.fit(self.data_normalized)
                suggested_point = block.optimize()
                self.measure(suggested_point)
            else:
                block.run()
                self.data = self.data.append(block.data)
