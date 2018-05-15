import dynet as dy
from tqdm import tqdm

from morphology import metrics


def nll(model, instance):
    state = model.initialize(instance)
    loss = []
    for y1, y2 in zip(instance.ys, instance.ys[1:]):
        prob_nodes, state = model(instance, y1, *state)
        loss.append(-dy.log(prob_nodes[y2]))
    return dy.esum(loss)


def total_loss(model, data):
    losses = []
    for instance in tqdm(data, desc='Computing loss'):
        losses.append(nll(model, instance).value())
        dy.renew_cg()
    return sum(losses)


def accuracy(data, predictions):
    predictions = [prediction for prediction, _ in predictions]
    expected = [instance.ys for instance in data]
    return (metrics.accuracy(predictions, expected) * 100,
            metrics.levenshtein_distance(predictions, expected))


def generate(model, searcher, data):
    predictions = []
    for instance in tqdm(data, desc='Generating predictions'):
        predictions.append(searcher.search(model, instance))
    return predictions


def oracle(data, predictions):
    correct = 0
    total = len(data)
    for instance, (_, beam) in zip(data, predictions):
        preds = [prediction for _, prediction in beam]
        correct += instance.ys in preds
    return correct / total * 100
