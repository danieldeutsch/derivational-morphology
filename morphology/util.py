import json

from morphology.models.seq2seq import evaluate


def save_predictions(predictions, vocab, filename):
    decoded = []
    for prediction, beam in predictions:
        prediction = ''.join([vocab[c] for c in prediction])
        decoded_beam = []
        for score, value in beam:
            value = ''.join([vocab[c] for c in value])
            decoded_beam.append((score, value))
        decoded.append((prediction, decoded_beam))

    json.dump(decoded, open(filename, 'w'))


def load_predictions(vocab, filename):
    decoded = json.load(open(filename, 'r'))
    encoded = []
    for prediction, beam in decoded:
        prediction = [vocab[c] for c in prediction]
        encoded_beam = []
        for score, value in beam:
            value = [vocab[c] for c in value]
            encoded_beam.append((score, value))
        encoded.append((prediction, encoded_beam))

    return encoded


def _filter_to_transformation(data, predictions, t):
    t_data, t_preds = [], []
    for instance, pred in zip(data, predictions):
        if instance.transformation == t:
            t_data.append(instance)
            t_preds.append(pred)
    return t_data, t_preds


def calculate_metrics(data, predictions):
    metrics = {}

    accuracy, edit = evaluate.accuracy(data, predictions)
    oracle = evaluate.oracle(data, predictions)
    metrics['acc'] = accuracy
    metrics['edit'] = edit
    metrics['top-k-acc'] = oracle

    for t in ['((ADJ-ADV))', '((VERB-NOM))', '((SUBJECT))', '((ADJ-NOM))']:
        t_data, t_preds = _filter_to_transformation(data, predictions, t)
        accuracy, edit = evaluate.accuracy(t_data, t_preds)
        oracle = evaluate.oracle(t_data, t_preds)
        metrics[f'acc-{t}'] = accuracy
        metrics[f'edit-{t}'] = edit
        metrics[f'top-k-acc-{t}'] = oracle

    return metrics


def save_metrics(metrics, model_name, split, filename):
    with open(filename, 'a') as out:
        for metric in sorted(metrics.keys()):
            value = metrics[metric]
            out.write(f'{model_name}\t{split}\t{metric}\t{value}\n')
