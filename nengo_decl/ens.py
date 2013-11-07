
from context import active_model

def _get(name):
    return active_model().get(name)


def ensemble(*args, **kwargs):
    """Create an ensemble"""
    active_model().make_ensemble(*args, **kwargs)


def encoders(name, val):
    """Assign encoders to `name` (typically an ensemble)
    """
    _get(name).encoders = val


def n_neurons(name):
    """Return the number of neurons of `name` (typically ensemble)
    """
    return _get(name).n_neurons


