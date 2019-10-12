import attr
from optimistic.pipeline import Pipeline

@attr.s
class Loop(Pipeline):
    ''' The Loop block is a subpipeline which can repeatedly execute a sequence
        of blocks which are added to it. For example, this could be used to
        implement a closed-loop learning cycle as follows:
        1. Fit a response surface to acquired data
        2. Numerically optimize to suggest the next experimental point
        3. Measure the suggested point, add to the dataset, and repeat from step 1
    '''
    loops = attr.ib(default=1, converter=int)

    def run(self):
        for i in range(self.loops):
            Pipeline.run(self)
