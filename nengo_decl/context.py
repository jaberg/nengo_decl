import nengo

model_stack = []

class declarative_syntax(object):
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        model_stack.append(self.model)

    def __exit__(self, *args):
        if [self.model] == model_stack[-1:]:
            model_stack.pop()


def active_model():
    return model_stack[-1]

