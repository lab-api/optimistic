
import attr
from optimistic.pipeline import Pipeline

@attr.s
class Loop(Pipeline):
    loops = attr.ib(default=1, converter=int)

    def run(self):
        for i in range(self.loops):
            Pipeline.run(self) 
