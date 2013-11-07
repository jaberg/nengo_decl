from context import declarative_syntax, active_model
from conn import connect
from ens import (
    ensemble,
    encoders,
    n_neurons,
    )

def node(*args, **kwargs):
    """Create a Node"""
    active_model().make_node(*args, **kwargs)

def probe(*args, **kwargs):
    """Create a Probe"""
    active_model().probe(*args, **kwargs)
