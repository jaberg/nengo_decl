
Nengo Declarative Syntax
========================

"Object-free" model definition (and backend-configuration) language for Nengo.


Simple example: multiplication
------------------------------


```python
import numpy as np
import nengo
import nengo.helpers
from nengo_decl import *

model = nengo.Model('multiplication')
with declarative_syntax(model):
    ensemble('A', nengo.LIF(100), dimensions=1, radius=10)
    ensemble('B', nengo.LIF(100), dimensions=1, radius=10)
    ensemble('Combined', nengo.LIF(100), dimensions=2, radius=15)
    ensemble('D', nengo.LIF(100), dimensions=1, radius=20)

    encoders('Combined',
         np.tile([[1,1],[-1,1],[1,-1],[-1,-1]],
             (n_neurons('Combined')/4, 1)))

    node('Input A', nengo.helpers.piecewise({0:0, 2.5:10, 4:-10}))
    node('Input B', nengo.helpers.piecewise({0:10, 1.5:2, 3:0, 4.5:2}))

    connect('Input A', 'A')
    connect('Input B', 'B')
    connect('A','Combined', transform=[[1], [0]])
    connect('B','Combined', transform=[[0], [1]])
    connect('Combined', 'D', function=lambda x: x[0] * x[1])

    for name in 'Input A', 'Input B':
        probe(name)

    for name in 'A', 'B', 'Combined', 'D':
        probe(name, filter=0.01)

sim = model.simulator()

sim.run(5)
```


More elaborate example
----------------------

See nengo_decl/Qnetworks.py.

I got a little bit bogged down where I didn't know the old API well enough to
translate it into new API concepts *and* new syntax, but I got some ways:

```python
net = nef.Network("QNetwork")
with declarative_syntax(net):
    N = 50
    statelength = math.sqrt(2*stateradius**2)
    tauPSC = 0.007
    num_actions = len(actions)
    init_Qs = 0.0
    weight_save = 600.0 #period to save weights (realtime, not simulation time)

    #set up relays
    direct_mode('state_relay', 1, dimension=stateD)
    add_decoded_termination('state_relay', 'input', MU.I(stateD), .001, False)

    #create state population
    ensemble('state_pop', neurons=LIF(stateN),
             dimensions=stateD,
             radius=statelength,
             encoders=state_encoders,
            )
    connect('state_relay', 'state_pop', filter=tauPSC)

    memory('saved_state', neurons=LIF(N * 4), dimension=stateD,
           inputscale=50,
           radius=stateradius,
           direct_storage=True)

    # N.B. the "." syntax refers to an ensemble created by the `memory` macro
    connect('state_relay', 'saved_state.target')

    ensemble('old_state_pop', neurons=LIF(stateN),
             dimensions=stateD,
             radius=statelength,
             encoders=state_encoders)

    connect('saved_state', 'old_state_pop', filter=tauPSC)

    # mess with the intercepts ?
    for name in 'state_pop', 'old_state_pop':
        set_intercepts(name, IndicatorPDF(0, 1))

    fixMode('state_relay')
    fixMode('state_pop', ['default', 'rate'])
    fixMode('old_state_pop', ['default', 'rate'])

    #set up action nodes

    #XXX getting bogged down because I don't know Java API well enough :(
```
