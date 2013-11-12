import nengo
#import nengo.networks

model_stack = []

class declarative_syntax(object):
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        model_stack.append(self.model)

    def __exit__(self, *args):
        if [self.model] == model_stack[-1:]:
            top = model_stack.pop()
        # XXX at what point should a submodel be added to the parent model
        #if model_stack:
            #model_stack[-1].objs.update(top.objs)


def active_model():
    return model_stack[-1]


def subnetwork(name):
    # -- XXX: figure out when add_to_model should be called
    #         so that this actually works.
    #return nengo.networks.Network(name)

    # -- XXX: abuse Model to be a sub-model as well (should be Network?)
    rval = nengo.Model(name)
    # -- XXX should immediately add to active model so that name lookup by
    #         parent works right away?
    return rval


