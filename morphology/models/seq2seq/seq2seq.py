import dynet as dy

from morphology.models.seq2seq import (BiLSTMEncoder, EmptyAttender,
                                       GeneralLinearAttender, MLPAttender,
                                       DecoderAction)

_initializer = dy.UniformInitializer(0.1)
_zero_initializer = dy.ConstInitializer(0.0)


class Seq2SeqModel(object):
    def __init__(self, pc, V, embed_dim, hidden_dim, num_layers=1, attention=None):
        self.embeddings = pc.add_lookup_parameters((V, embed_dim), _initializer)

        self.encoder = BiLSTMEncoder(pc, V, embed_dim, hidden_dim, num_layers,
                                     self.embeddings)
        self.decoder = dy.VanillaLSTMBuilder(1, embed_dim, hidden_dim, pc)
        self.decoder_action = DecoderAction(pc, V, hidden_dim)

        # Set the type of attender
        if attention == 'linear':
            self.attender = GeneralLinearAttender(hidden_dim, hidden_dim, pc)
        elif attention == 'mlp':
            self.attender = MLPAttender(hidden_dim, hidden_dim, hidden_dim, pc)
        elif attention is None:
            self.attender = EmptyAttender()
        else:
            raise Exception('Unknown attention type: ' + attention)

    def initialize(self, instance):
        encoder_states, last_hidden = self.encoder(instance)

        decoder_state = self.decoder.initial_state()
        decoder_state = decoder_state.set_s(last_hidden)

        return encoder_states, decoder_state

    def __call__(self, instance, y, encoder_states, decoder_state):
        y = self.embeddings[y]
        decoder_state = decoder_state.add_input(y)
        hidden_layer = self.attender(decoder_state.output(), encoder_states)

        # Compute the probability over the vocabulary for the next step
        prob_nodes = self.decoder_action(instance, y, hidden_layer)

        return prob_nodes, (encoder_states, decoder_state)
