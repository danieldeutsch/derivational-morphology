import dynet as dy
import heapq
import numpy as np

from morphology.models.seq2seq.search.searcher import Searcher


class TopKShortestPathSearch(Searcher):
    def __init__(self, K, bos, eos, max_output_len=20):
        super().__init__()
        self.K = K
        self.bos = bos
        self.eos = eos
        self.max_output_len = max_output_len

    def search(self, model, instance):
        dy.renew_cg()
        self.next_example()

        state = model.initialize(instance)
        queue = []
        heapq.heappush(queue, (0, [self.bos], state))
        outputs = []

        while queue:
            self.next_state()
            loss, ys, state = heapq.heappop(queue)
            if self._is_valid_output(ys):
                outputs.append((loss, ys))
                if len(outputs) == self.K:
                    best_ys = outputs[0][1]
                    return best_ys, outputs
            else:
                prob_nodes, next_state = model(instance, ys[-1], *state)
                (V,), _ = prob_nodes.dim()
                allowed_ys = self._get_allowed_ys(ys, V)
                for y in allowed_ys:
                    next_ys = ys + [y]
                    next_loss = loss - np.log(prob_nodes[y].value())
                    heapq.heappush(queue, (next_loss, next_ys, next_state))

        raise Exception('Could not find top k shortest paths.')

    def _get_allowed_ys(self, ys, V):
        return range(V)

    def _is_valid_output(self, ys):
        if len(ys) == self.max_output_len:
            return True
        if (ys[-1] == self.eos and len(ys) > 1):
            return True
        return False


class ShortestPathSearch(TopKShortestPathSearch):
    def __init__(self, bos, eos, max_output_len=20):
        super().__init__(1, bos, eos, max_output_len=max_output_len)


class ApproximateShortestPathSearch(Searcher):
    def __init__(self, K, bos, eos):
        super().__init__()
        self.K = K
        self.bos = bos
        self.eos = eos

    def search(self, model, instance):
        dy.renew_cg()
        self.next_example()

        state = model.initialize(instance, training_mode=False)
        queue = []
        outputs = []
        heapq.heappush(queue, (0, [self.bos], state))

        while queue:
            self.next_state()
            loss, ys, state = heapq.heappop(queue)
            prob_nodes, next_state = model(instance, ys[-1], *state)

            (V,), _ = prob_nodes.dim()
            allowed_ys = self._get_allowed_ys(ys, V)
            for y in allowed_ys:
                next_ys = ys + [y]
                next_loss = loss - np.log(prob_nodes[y].value())
                if self._is_valid_output(next_ys):
                    outputs.append((next_loss, next_ys))
                else:
                    heapq.heappush(queue, (next_loss, next_ys, next_state))

            if len(outputs) >= self.K:
                break

        sorted_outputs = sorted(outputs, key=lambda t: t[0])
        prediction = sorted_outputs[0][1]
        return prediction, sorted_outputs

    def _get_allowed_ys(self, ys, V):
        return range(V)

    def _is_valid_output(self, ys):
        if (ys[-1] == self.eos and len(ys) > 1):
            return True
        return False
