import unittest
import numpy as np

import nengo
import nengo.helpers
from nengo.objects import Uniform
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

            ens.node('Input A', nengo.helpers.piecewise(
                {0:0, 2.5:10, 4:-10}))
            ens.node('Input B', nengo.helpers.piecewise(
                {0:10, 1.5:2, 3:0, 4.5:2}))

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
        plt.show()
        



    def test_basal_ganglia(self):
        # connection weights from (Gurney, Prescott, & Redgrave, 2001)
        mm = 1
        mp = 1
        me = 1
        mg = 1
        ws = 1
        wt = 1
        wm = 1
        wg = 1
        wp = 0.9
        we = 0.3
        e = 0.2
        ep = -0.25
        ee = -0.2
        eg = -0.2
        le = 0.2
        lg = 0.2

        dimensions = 1
        n_neurons_per_ensemble = 100
        radius = 1.5
        tau_ampa = 0.002
        tau_gaba = 0.008
        output_weight = -3

        encoders = np.ones((n_neurons_per_ensemble, 1))

        model = nengo.Model('Basal Ganglia')
        with declarative_syntax(model):
            for name, lbound in (
                ('StrD1', e),
                ('StrD2', e),
                ('STN', ep),
                ('GPi', eg),
                ('GPe', ee)):
                ens.ensemble_array(name,
                               intercepts=Uniform(lbound, 1),
                               neurons= nengo.LIF(
                                   n_neurons_per_ensemble * dimensions),
                               n_ensembles= dimensions,
                               radius= radius,
                               encoders= encoders,
                              )
            ens.passthrough('input', dimensions=dimensions)
            ens.passthrough('output', dimensions=dimensions)

            # spread the input to StrD1, StrD2, and STN
            conn.connect('input', 'StrD1',
                    filter=None, 
                    transform=np.eye(dimensions) * ws * (1 + lg))

            conn.connect('input', 'strD2',
                    filter=None,
                    transform=np.eye(dimensions) * ws * (1 - le))

            conn.connect('input', 'STN',
                    filter=None,
                    transform=np.eye(dimensions) * wt)

            # connect the striatum to the GPi and GPe (inhibitory)
            def func_str(x):
                return max(x[0] - e, 0) * mm

            conn.connect('StrD1', 'GPi',
                    function=func_str,
                    filter=tau_gaba,
                    transform=-np.eye(dimensions) * wm)

            conn.connect('StrD2', 'GPe',
                    function=func_str,
                    filter=tau_gaba,
                    transform=-np.eye(dimensions) * wm)
                        
            # connect the STN to GPi and GPe (broad and excitatory)
            def func_stn(x):
                return max(x[0] - ep) * mp

            tr = np.ones((dimensions, dimensions)) * wp
            conn.connect('STN', 'GPi',
                    function=func_stn,
                    transform=tr,
                    filter=tau_ampa)
            conn.connect('STN', 'GPe',
                    function=func_stn,
                    transform=tr,
                    filter=tau_ampa)

            # connect the GPe to GPi and STN (inhibitory)
            def func_gpe(x):
                return max(x[0] - ee) * me
            conn.connect('GPe', 'GPi',
                    function=func_gpe,
                    filter=tau_gaba,
                    transform=-np.eye(dimensions) * we)
            conn.connect('GPe', 'STN',
                    function=func_gpe,
                    filter=tau_gaba,
                    transform=-np.eye(dimensions) * wg)

            #connect GPi to output (inhibitory)
            conn.connect('GPi', 'output',
                    function=lambda x: max(x[0] - eg) * mg,
                    filter=None,
                    transform=np.eye(dimensions) * output_weight)

