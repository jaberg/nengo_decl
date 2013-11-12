
import numpy as np

import nengo
from nengo.objects import Uniform

from nengo_decl import conn, ens, declarative_syntax, subnetwork


def BG(name,
       dimensions = 1,
       n_neurons_per_ensemble = 100,
       radius = 1.5,
       model=None,
       mm = 1,
       mp = 1,
       me = 1,
       mg = 1,
       ws = 1,
       wt = 1,
       wm = 1,
       wg = 1,
       wp = 0.9,
       we = 0.3,
       e = 0.2,
       ep = -0.25,
       ee = -0.2,
       eg = -0.2,
       le = 0.2,
       lg = 0.2,
       tau_ampa = 0.002,
       tau_gaba = 0.008,
       output_weight = -3,
      ):
    # connection weights from (Gurney, Prescott, & Redgrave, 2001)

    if model is None:
        # -- create a subnetwork of the active model
        #    (see nengo_decl.context.active_model)
        model = subnetwork(name)

    with declarative_syntax(model):
        encoders = np.ones((n_neurons_per_ensemble, 1))
        for label, lbound in (
            ('StrD1', e),
            ('StrD2', e),
            ('STN', ep),
            ('GPi', eg),
            ('GPe', ee)):
            ens.ensemble_array(label,
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
    return model
