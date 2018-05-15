import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm

from morphology.vocab import BOS


def get_suffixes(data, min_count=0):
    """Generates the set of possible suffixes in the training data. A
    suffix is defined by taking the longest prefix of x which is also a
    prefix of y.
    """
    suffixes = Counter()
    for item in data:
        x, y = item.source, item.target
        for i in range(len(x), 0, -1):
            prefix = x[:i]
            if y.startswith(prefix):
                suffix = y[len(prefix):]
                if len(suffix) > 0:
                    suffixes[suffix] += 1
                break
    return [suffix for suffix, count in suffixes.items() if count >= min_count]


class PrefixTree(object):
    def __init__(self, count_file, vocab, min_log_count, train=None):
        self.vocab = vocab
        self.raw_counts = self._load_raw_counts(count_file)
        self._filter_by_freq(min_log_count)
        if train is not None:
            self._filter_by_suffix(train)
        self._filter_by_vocab()
        self.prefix_counts = self._load_prefix_counts()
        self.next_states_cache = {}

    def _load_raw_counts(self, filename):
        raw_counts = {}
        with open(filename, 'r') as f:
            for line in tqdm(f, desc='Loading ngram counts'):
                token, count = line.strip().split('\t')
                if token[0].islower():
                    raw_counts[token] = int(count)
        return raw_counts

    def _filter_by_freq(self, min_log_count):
        tokens = list(self.raw_counts.keys())
        for token in tqdm(tokens, desc='Filtering raw counts based on min count'):
            if self.raw_counts[token] < np.exp(min_log_count):
                del self.raw_counts[token]

        initial = len(tokens)
        remaining = len(self.raw_counts)
        removed = initial - remaining
        percent = removed / initial * 100
        print('Deleted {} tokens ({:.2f}%)'.format(removed, percent))

    def _filter_by_suffix(self, data):
        suffixes = get_suffixes(data)
        tokens = list(self.raw_counts.keys())
        for token in tqdm(tokens, desc='Filtering raw counts based on suffix'):
            ok = False
            for suffix in self._suffixes(token):
                if suffix in suffixes:
                    ok = True
                    break
            if not ok:
                del self.raw_counts[token]

        initial = len(tokens)
        remaining = len(self.raw_counts)
        removed = initial - remaining
        percent = removed / initial * 100
        print('Deleted {} tokens ({:.2f}%)'.format(removed, percent))

    def _prefixes(self, token):
        for i in range(1, len(token) + 1):
            yield token[:i]

    def _suffixes(self, token):
        for i in range(len(token), 0, -1):
            yield token[i:]

    def _filter_by_vocab(self):
        tokens = list(self.raw_counts.keys())
        for token in tqdm(tokens, desc='Filtering based on vocab'):
            for char in token:
                if char not in self.vocab:
                    del self.raw_counts[token]
                    break

        initial = len(tokens)
        remaining = len(self.raw_counts)
        removed = initial - remaining
        percent = removed / initial * 100
        print('Deleted {} tokens ({:.2f}%)'.format(removed, percent))

    def _load_prefix_counts(self):
        prefix_counts = defaultdict(int)
        for token, count in tqdm(self.raw_counts.items(),
                                 total=len(self.raw_counts),
                                 desc='Building the prefix tree'):
            for prefix in self._prefixes(BOS + token + BOS):
                prefix_counts[prefix] += count

        for token, count in tqdm(prefix_counts.items(),
                                 total=len(prefix_counts),
                                 desc='Converting to log counts'):
            prefix_counts[token] = np.log(count)
        return prefix_counts

    def count(self, token):
        try:
            return self.raw_counts[token]
        except KeyError:
            return 0

    def next_states(self, token):
        if token in self.next_states_cache:
            return self.next_states_cache[token]
        else:
            states = []
            for i, w in enumerate(self.vocab.id2tkn):
                if (token + w) in self.prefix_counts:
                    states.append(i)
            self.next_states_cache[token] = states
            return states
