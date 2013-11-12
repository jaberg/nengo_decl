import unittest
import numpy as np

import nengo
from nengo.helpers import piecewise
import demo_bg
from nengo_decl import conn, ens, probe, declarative_syntax

class TestBasics(unittest.TestCase):
    def test_multiplication(self):
        model = nengo.Model('multiplication')

        with declarative_syntax(model):
            ens.ensemble('A', nengo.LIF(100), dimensions=1, radius=10)
            ens.ensemble('B', nengo.LIF(100), dimensions=1, radius=10)
            ens.ensemble('Combined', nengo.LIF(100), dimensions=2, radius=15)
            ens.ensemble('D', nengo.LIF(100), dimensions=1, radius=20)

            ens.encoders('Combined',
                 np.tile([[1,1],[-1,1],[1,-1],[-1,-1]],
                     (ens.n_neurons('Combined')/4, 1)))

            ens.node('Input A', piecewise({0:0, 2.5:10, 4:-10}))
            ens.node('Input B', piecewise({0:10, 1.5:2, 3:0, 4.5:2}))

            conn.connect('Input A', 'A')
            conn.connect('Input B', 'B')
            conn.connect('A','Combined', transform=[[1], [0]])
            conn.connect('B','Combined', transform=[[0], [1]])
            conn.connect('Combined', 'D', function=lambda x: x[0] * x[1])

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
        #plt.show()
        
    def test_basal_ganglia(self):
        # --smoke test that it runs
        bg = demo_bg.BG()

    def test_nested_network(self):
        # --smoke test that it runs
        base = nengo.Model('base')
        with declarative_syntax(base):
            demo_bg.BG('BG1')
            demo_bg.BG('BG2')

            # -- this doesn't work yet, but I think it should
            conn.connect('BG1.output', 'BG1.input')

