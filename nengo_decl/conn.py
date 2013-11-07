

from context import active_model

def _get(name):
    return active_model().get(name)


def connect(*args, **kwargs):
    active_model().connect(*args, **kwargs)

