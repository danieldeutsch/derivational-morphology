import dynet as dy


class ConstraintLoss(object):
    def __init__(self, ptree):
        self.ptree = ptree

    def __call__(self, model, instance):
        state = model.initialize(instance)
        loss = []
        for i, (y1, y2) in enumerate(zip(instance.ys, instance.ys[1:])):
            prob_nodes, state = model(instance, y1, *state)

            prefix = ''.join(instance.raw_ys[:i+1])
            allowed_y = self.ptree.next_states(prefix)
            if y2 not in allowed_y:
                loss.append(-dy.log(prob_nodes[y2]))
            else:
                log_probs = dy.log_softmax(prob_nodes, restrict=allowed_y)
                loss.append(-log_probs[y2])
        return dy.esum(loss)


class NegativeLogLikelihoodLoss(object):
    def __init__(self):
        pass

    def __call__(self, model, instance):
        state = model.initialize(instance)
        loss = []
        for y1, y2 in zip(instance.ys, instance.ys[1:]):
            prob_nodes, state = model(instance, y1, *state)
            loss.append(-dy.log(prob_nodes[y2]))
        return dy.esum(loss)
