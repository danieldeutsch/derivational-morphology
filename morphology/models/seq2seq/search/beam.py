import dynet as dy
import numpy as np

from morphology.models.seq2seq.search.searcher import Searcher


class BeamSearch(Searcher):
    def __init__(self, K, bos, eos, max_output_len=50):
        super().__init__()
        self.K = K
        self.bos = bos
        self.eos = eos
        self.max_output_len = max_output_len

    def search(self, model, instance):
        dy.renew_cg()
        self.next_example()

        state = model.initialize(instance)
        beam = [(0, [self.bos], state)]
        outputs = []
        while len(outputs) < self.K:
            next_beam = []
            for loss, ys, state in beam:
                self.next_state()
                if (ys[-1] == self.eos and len(ys) > 1) or len(ys) == self.max_output_len:
                    outputs.append((loss, ys))
                    continue

                prob_nodes, next_state = model(instance, ys[-1], *state)

                probs = prob_nodes.value()
                (V,), _ = prob_nodes.dim()
                allowed_ys = self._get_allowed_ys(ys, V)
                sorted_probs = sorted([(probs[y], y) for y in allowed_ys], key=lambda t: -t[0])

                for prob, v in sorted_probs[:self.K - len(outputs)]:
                    next_loss = loss - np.log(prob)
                    next_ys = ys + [v]
                    next_beam.append((next_loss, next_ys, next_state))

            beam = sorted(next_beam, key=lambda t: t[0])[:self.K - len(outputs)]

        sorted_outputs = sorted(outputs, key=lambda t: t[0])
        prediction = sorted_outputs[0][1]
        return prediction, sorted_outputs

    def _get_allowed_ys(self, ys, V):
        return range(V)
