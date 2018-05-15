import dynet as dy
from dynet import parameter, tanh, affine_transform, concatenate, transpose, softmax, esum, cmult

_initializer = dy.UniformInitializer(0.1)
_zero_initializer = dy.ConstInitializer(0.0)


class Attender(object):
    '''An attention module; for generating context vectors in decoding.'''
    def __init__(self, decoder_dim, encoder_dim, pc):
        '''Wc is the weight matrix for transforming from concatenated space to
        final features. b is the bias term for transforming from concatenated
        space to final features.
        '''
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.Wc = pc.add_parameters((decoder_dim, encoder_dim + decoder_dim), _initializer)
        self.b = pc.add_parameters((decoder_dim), _zero_initializer)

    def _ordered_pairwise_affinity(main, auxiliary):
        '''Computes the affinity of two vectors.'''
        raise NotImplementedError

    def _get_context_vector(self, decoder_state, encoder_states):
        '''Returns the weighted average of the encoder state outputs
        according to affinity with the decoder state output.
        '''
        encoder_outputs = [concatenate([x[0].output(), x[1].output()])
                for x in encoder_states]
        affinity_vector = [self._ordered_pairwise_affinity(decoder_state, x) for x in encoder_outputs]
        alignment_vector = softmax(concatenate(affinity_vector))
        context_vector = esum([cmult(enc_state, a_prob) for enc_state, a_prob
            in zip(alignment_vector, encoder_outputs)])
        return context_vector

    def __call__(self, decoder_state, encoder_states, *args):
        '''Computes the context vector over encoder states for a decoder state.'''
        pWc = parameter(self.Wc)
        pb = parameter(self.b)
        context = self._get_context_vector(decoder_state, encoder_states)
        prediction_feature_vector = affine_transform([pb, pWc, concatenate([decoder_state, context])])
        return prediction_feature_vector


class EmptyAttender(Attender):
    def __init__(self):
        pass

    def __call__(self, decoder_state, encoder_states, *args):
        return decoder_state.output()


class GeneralLinearAttender(Attender):
    def __init__(self, decoder_dim, encoder_dim, pc):
        super().__init__(decoder_dim, encoder_dim, pc)
        self.W_linear = pc.add_parameters((decoder_dim, encoder_dim), _initializer)

    def _ordered_pairwise_affinity(self, main, auxiliary):
        W = parameter(self.W_linear)
        affinity = (transpose(main) * W) * auxiliary
        return affinity


class MLPAttender(Attender):
    def __init__(self, decoder_dim, encoder_dim, attention_dim, pc):
        super().__init__(decoder_dim, encoder_dim, pc)
        self.W_mlp = pc.add_parameters((attention_dim, encoder_dim + decoder_dim), _initializer)
        self.b_mlp = pc.add_parameters((attention_dim), _zero_initializer)
        self.v_mlp = pc.add_parameters((attention_dim), _initializer)

    def _ordered_pairwise_affinity(self, decoder_state, encoder_state):
        W = parameter(self.W_mlp)
        b = parameter(self.b_mlp)
        v = parameter(self.v_mlp)

        concat = concatenate([decoder_state, encoder_state])
        layer = tanh(affine_transform([b, W, concat]))
        affinity = transpose(v) * layer
        return affinity
