import dynet as dy

_initializer = dy.UniformInitializer(0.1)
_zero_initializer = dy.ConstInitializer(0.0)


class DecoderAction(object):
    def __init__(self, pc, V, hidden_dim, num_layers=1, embeddings=None):
        self.embeddings = embeddings
        self.R = pc.add_parameters((V, hidden_dim), _initializer)
        self.bias = pc.add_parameters((V), _zero_initializer)

    def __call__(self, instance, y, hidden_layer):
        R = dy.parameter(self.R)
        bias = dy.parameter(self.bias)
        prob_nodes = dy.softmax(dy.affine_transform([bias, R, hidden_layer]))
        return prob_nodes
