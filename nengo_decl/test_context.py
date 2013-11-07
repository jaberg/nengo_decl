import unittest
import numpy as np

import nengo
import nengo.helpers
from nengo_decl import *

class TestBasics(unittest.TestCase):
    def test_multiplication(self):
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

        import matplotlib.pyplot as plt

        # Plot the input signals and decoded ensemble values
        correct_ans = nengo.helpers.piecewise({0:0, 1.5:0, 2.5:20, 3:0, 4:0, 4.5:-20})
        t = sim.data(model.t)
        plt.plot(t, sim.data('A'), label="Decoded A")
        plt.plot(t, sim.data('B'), label="Decoded B")
        plt.plot(t, sim.data('D'), label="Decoded D")
        out = [0] * t.shape[0]
        for i in np.arange(t.shape[0]):
            out[i] = correct_ans(t[i])
        plt.plot(t, out)
        plt.legend()
        plt.ylim(-25,25);
        plt.show()
        



