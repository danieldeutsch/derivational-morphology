import dynet as dy

_initializer = dy.UniformInitializer(0.1)


class BiLSTMEncoder(object):
    def __init__(self, pc, V, embed_dim, hidden_dim, num_layers=1, embeddings=None):
        self.embeddings = embeddings or pc.add_lookup_parameters((V, embed_dim), _initializer)
        self.encoder = dy.BiRNNBuilder(num_layers, embed_dim, hidden_dim, pc, dy.VanillaLSTMBuilder)

    def __call__(self, instance):
        xs = [self.embeddings[x] for x in instance.xs]
        encoder_states = self.encoder.add_inputs(xs)

        last_encoder_tuple = encoder_states[-1]
        last_encoder_mem = dy.concatenate([last_encoder_tuple[0].s()[0], last_encoder_tuple[1].s()[0]])
        last_encoder_cell = dy.concatenate([last_encoder_tuple[0].s()[1], last_encoder_tuple[1].s()[1]])

        return encoder_states, (last_encoder_mem, last_encoder_cell)
