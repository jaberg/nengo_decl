from context import declarative_syntax, active_model
from conn import connect
from ens import (
    ensemble,
    ensemble_array,
    encoders,
    n_neurons,
    node,
    passthrough,
    )


def probe(*args, **kwargs):
    """Create a Probe"""
    active_model().probe(*args, **kwargs)
