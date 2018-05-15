import argparse
import gensim
import numpy
import sys

import morphology.instance

BOS = '#'
UNK = 'UNK'


class Vocab(object):
    def __init__(self, bos=True, unk=True):
        self.id2tkn = []
        self.tkn2id = {}
        self.closed = False
        if bos: self.add(BOS)
        if unk: self.add(UNK)

    def add(self, item):
        if type(item) is list:
            return [self._add(token) for token in item]
        return self._add(item)

    def _add(self, token):
        assert not self.closed
        try:
            return self.tkn2id[token]
        except KeyError:
            id_ = len(self)
            self.id2tkn.append(token)
            self.tkn2id[token] = id_
            return id_

    def close(self):
        self.closed = True

    def __getitem__(self, key):
        if type(key) is str:
            try:
                return self.tkn2id[key]
            except KeyError:
                return self.tkn2id[UNK]
        elif type(key) is int:
            return self.id2tkn[key]
        else:
            raise KeyError(key)

    def __len__(self):
        return len(self.id2tkn)

    @staticmethod
    def load(filename):
        vocab = Vocab()
        with open(filename, 'r') as f:
            for line in f:
                vocab.add(line.strip())
        vocab.close()
        return vocab


class DistributionalVocab:

    def __init__(self, path_to_vecs):
        self.path_to_vecs = path_to_vecs
        print("Loading word2vec vectors...", file=sys.stderr)
        if path_to_vecs == 'empty':
            self.vector_model = {}
        else:
            self.vector_model = gensim.models.KeyedVectors.load_word2vec_format(
                    path_to_vecs, binary=True)
        self.VECTOR_DIM = 300

    def add_from_file(self, vecs_path, instances):
        self.vector_model.update(
                {instance.source:vec for vec, instance in zip(
                    numpy.loadtxt(vecs_path), instances)}
                )

    def __getitem__(self, key, default='empty'):
        if key in self.vector_model:
            return self.vector_model[key]
        else:
            if default == 'error':
                raise KeyError("Looked up word not present in vectors")
            elif default == 'empty':
                print("WARNING: Empty vector loaded for {}".format(key))
                #return numpy.random.normal(size=300)
                return numpy.zeros(300)
            else:
                raise ValueError("Unknown default vector return used in distributional vocab")


def main(args):
    data = morphology.instance.Instance.load(args.input_src, args.input_trg)

    vocab = Vocab()
    for instance in data:
        for c in instance.raw_xs + instance.raw_ys:
            vocab.add(c)

    with open(args.output_file, 'w') as out:
        for c in vocab.id2tkn:
            out.write(c + '\n')


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--input-src', required=True)
    argp.add_argument('--input-trg', required=True)
    argp.add_argument('--output-file', required=True)
    args = argp.parse_args()
    main(args)
