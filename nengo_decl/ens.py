
from context import active_model

from nengo.templates import EnsembleArray
from nengo.objects import PassthroughNode


def _get(name):
    return active_model().get(name)


def ensemble(*args, **kwargs):
    """Create an Ensemble"""
    active_model().make_ensemble(*args, **kwargs)


def ensemble_array(*args, **kwargs):
    """Create an EnsembleArray"""
    active_model().add(EnsembleArray(*args, **kwargs))


def encoders(name, val):
    """Assign encoders to `name` (typically an ensemble)
    """
    _get(name).encoders = val


def n_neurons(name):
    """Return the number of neurons of `name` (typically ensemble)
    """
    return _get(name).n_neurons


def node(*args, **kwargs):
    """Create a Node"""
    active_model().make_node(*args, **kwargs)

def passthrough(*args, **kwargs):
    """Create a Passthrough Node"""
    active_model().add(PassthroughNode(*args, **kwargs))

