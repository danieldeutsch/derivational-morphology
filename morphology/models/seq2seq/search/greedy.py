from morphology.models.seq2seq.search import BeamSearch


class GreedySearch(BeamSearch):
    def __init__(self, bos, eos, max_output_len=50):
        super().__init__(1, bos, eos, max_output_len)
