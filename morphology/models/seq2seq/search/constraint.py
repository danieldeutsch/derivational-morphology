from morphology.models.seq2seq.search import (ApproximateShortestPathSearch,
                                              BeamSearch, ShortestPathSearch,
                                              TopKShortestPathSearch)


class ConstraintBeamSearch(BeamSearch):
    def __init__(self, K, bos, eos, ptree, vocab, **kwargs):
        super().__init__(K, bos, eos, **kwargs)
        self.ptree = ptree
        self.vocab = vocab

    def _get_allowed_ys(self, ys, V):
        token = ''.join([self.vocab[y] for y in ys])
        return self.ptree.next_states(token)


class ConstraintGreedySearch(ConstraintBeamSearch):
    def __init__(self, bos, eos, ptree, vocab, **kwargs):
        super().__init__(1, bos, eos, ptree, vocab, **kwargs)


class ConstraintTopKShortestPathSearch(TopKShortestPathSearch):
    def __init__(self, K, bos, eos, ptree, vocab, **kwargs):
        super().__init__(K, bos, eos, **kwargs)
        self.ptree = ptree
        self.vocab = vocab

    def _get_allowed_ys(self, ys, V):
        token = ''.join([self.vocab[y] for y in ys])
        return self.ptree.next_states(token)


class ConstraintShortestPathSearch(ShortestPathSearch):
    def __init__(self, bos, eos, ptree, vocab, **kwargs):
        super().__init__(bos, eos, **kwargs)
        self.ptree = ptree
        self.vocab = vocab

    def _get_allowed_ys(self, ys, V):
        token = ''.join([self.vocab[y] for y in ys])
        return self.ptree.next_states(token)


class ConstraintApproximateShortestPathSearch(ApproximateShortestPathSearch):
    def __init__(self, K, bos, eos, ptree, vocab):
        super().__init__(K, bos, eos)
        self.ptree = ptree
        self.vocab = vocab

    def _get_allowed_ys(self, ys, V):
        token = ''.join([self.vocab[y] for y in ys])
        return self.ptree.next_states(token)
