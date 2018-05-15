import sys
import math
from nltk.metrics import edit_distance


def accuracy(observed, expected):
    """Computes the exact match accuracy."""
    assert len(observed) == len(expected), str(len(observed)) + ',' + str(len(expected))

    count = 0
    for x, y in zip(observed, expected):
        count += (x == y)
    return count / len(observed)


def levenshtein_distance(observed, expected):
    """Computes the average Levenshtein distance."""
    assert len(observed) == len(expected)

    total = 0
    for x, y in zip(observed, expected):
        total += edit_distance(x, y)
    return total / len(observed)


_suffixes = ['ly', 'er', 'ation', 'or', 'ity', 'ment', 'ist', 'ness', 'ence', 'ure', 'ee', 'age']
def suffix_f1(observed, expected):
    """Computes precision, recall, and f1 for every suffix defined in '_suffixes'."""
    assert len(observed) == len(expected)

    suffix_scores = {}
    for suffix in _suffixes:
        tp, fp, fn = 0, 0, 0
        for x, y in zip(observed, expected):
            if x.endswith(suffix) and y.endswith(suffix):
                tp += 1
            elif x.endswith(suffix) and not y.endswith(suffix):
                fp += 1
            elif not x.endswith(suffix) and y.endswith(suffix):
                fn += 1

        p = tp / (tp + fp) if (tp + fp) != 0 else math.nan
        r = tp / (tp + fn) if (tp + fn) != 0 else math.nan
        f1 = 2 * (p * r) / (p + r)
        suffix_scores[suffix] = {'p': p, 'r': r, 'f1': f1}

    return suffix_scores

if __name__ == '__main__':
  print(accuracy([x.strip() for x in open(sys.argv[1])], [x.strip() for x in open(sys.argv[2])] ))
