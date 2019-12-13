# optimistic
[![Build Status](https://travis-ci.org/lab-api/optimistic.svg?branch=master)](https://travis-ci.org/lab-api/optimistic)
[![Test Coverage](https://api.codeclimate.com/v1/badges/afd6fdffa20e92d9d08e/test_coverage)](https://codeclimate.com/github/lab-api/optimistic/test_coverage)
[![Maintainability](https://api.codeclimate.com/v1/badges/afd6fdffa20e92d9d08e/maintainability)](https://codeclimate.com/github/lab-api/optimistic/maintainability)

Optimistic is an optimization library for efficient exploration of multidimensional parameter spaces. It is designed to integrate with the [Parametric](https://github.com/lab-api/parametric) library to allow object-oriented representation and control of simulations or experiments.

## Simulation optimization
Let's say we want to optimize the restoring force of a magneto-optical trap. We can simulate the physics using the [MOTorNOT](https://github.com/robertfasano/motornot)
 library:

```python
%matplotlib inline
import numpy as np
from parametric import Attribute, parametrize
from optimistic import experiment
from MOTorNOT import SixBeam, LinearQuadrupole

linewidth = 2*np.pi*29e6

@parametrize
class MOT:
    power = Attribute('power', 10e-3)
    radius = Attribute('radius', 5e-3)
    detuning = Attribute('detuning', -1.6*linewidth)
    field_gradient = Attribute('field_gradient', 40e-2)
    
    @experiment
    def trapping(self):
        X = [self.radius()/2, 0, 0]      # atom halfway to the edge of the trap
        V = [0, 0, 0]    # atom at rest
        
        mot = SixBeam(power=self.power(), 
                      radius=self.radius(), 
                      detuning=self.detuning(), 
                      handedness=-1, 
                      field=LinearQuadrupole(self.field_gradient()).field)
        
        return mot.acceleration(X, V)[0, 0]   # trapping acceleration along x
    
mot = MOT()
```
We can investigate the variation of the trapping force with one or more parameters using a grid search. Let's see how the detuning affects the force:

```python
from optimistic.algorithms import GridSearch
results = GridSearch.study(mot.trapping, mot.detuning, (-5*linewidth, 0), steps=100)
```
![Detuning optimization](https://github.com/lab-api/optimistic/blob/master/tutorials/detuning_variation.png)

Now let's examine how the trapping force varies with the magnetic field gradient:

```python
results = GridSearch.study(mot.trapping, mot.field_gradient, (20e-2, 80e-2), steps=100)
```
![Gradient optimization](https://github.com/lab-api/optimistic/blob/master/tutorials/gradient_variation.png)

These two parameters are physically coupled by the effective detuning in the objective function; in other words, we would expect the optimal detuning to depend on the field gradient. Let's run a 2D search to see if this is true:

```python
grid = GridSearch(mot.trapping, steps=100)
grid.add_parameter(mot.detuning, (-5*linewidth, 0))
grid.add_parameter(mot.field_gradient, points=[20e-2, 40e-2, 60e-2, 80e-2])
grid.run()

import matplotlib.pyplot as plt
for x0, df in grid.data.groupby('field_gradient'):
    plt.plot(df['detuning'], df['trapping'], label=f'gradient={np.round(x0, 3)}')
plt.legend(loc=(1.04,0))
plt.xlabel('detuning')
plt.ylabel('trapping')
```
![Pair optimization](https://github.com/lab-api/optimistic/blob/master/tutorials/2d_variation.png)

We see that our predictions were right - as the field gradient increases, the optimal detuning also increases. Note that after running the 2D search, the optimizer returned to the best identified point, corresponding to about 200 MHz detuning and 80 G/cm gradient.

Using this optimization framework with a real experiment instead of a simulation is easy - we would simply provide "set_cmd" functions to the Attributes to talk to our devices, then use an objective function returning a physically measured quantity, like atomic fluorescence represented as the voltage on a photomultiplier tube.

