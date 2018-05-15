from morphology.vocab import BOS, Vocab, UNK


_allowed_transformations = ['VERB-NOM', 'ADJ-ADV', 'SUBJECT', 'OBJECT',
                            'ADJ-NOM']


class Instance(object):
    def __init__(self, raw_xs, raw_ys):
        self.raw_xs = raw_xs
        self.raw_ys = raw_ys
        self.xs = None
        self.ys = None
        self.xs_distr_vec = None
        self.ys_distr_vec = None
        self.xs_transf_vec = None

    def _clean_transformation(self, raw_xs):
        transformation = raw_xs[-1]
        for t in _allowed_transformations:
            if t in transformation:
                return raw_xs[:-1] + [t]
        return raw_xs[:-1] + [UNK]

    @property
    def source(self):
        return ''.join(self.raw_xs[1:-2])

    @property
    def target(self):
        return ''.join(self.raw_ys[1:-1])

    @property
    def transformation(self):
        return self.raw_xs[-1]

    def __repr__(self):
        return str(self)

    def __str__(self):
        return ''.join(self.raw_xs) + ' -> ' + ''.join(self.raw_ys)

    @staticmethod
    def load(src_file, trg_file):
        instances = []
        with open(src_file, 'r') as src_f, open(trg_file, 'r') as trg_f:
            for src, trg in zip(src_f, trg_f):
                src, trg = src.strip(), trg.strip()
                index = src.find('((')
                assert index > 0

                xs = [BOS] + src[:index-1].split() + [BOS] + [src[index:]]
                ys = [BOS] + trg.split() + [BOS]
                instance = Instance(xs, ys)
                instances.append(instance)

        return instances

    @staticmethod
    def build_vocab(instances):
        vocab = Vocab()
        for instance in instances:
            for c in instance.raw_xs + instance.raw_ys:
                vocab.add(c)
        vocab.close()
        return vocab

    @staticmethod
    def encode(instances, vocab):
        for instance in instances:
            instance.xs = [vocab[x] for x in instance.raw_xs]
            instance.ys = [vocab[y] for y in instance.raw_ys]

    @staticmethod
    def add_distr(instances, dist_vocab):
        for instance in instances:
            instance.xs_distr_vec = dist_vocab[instance.source]
            instance.ys_distr_vec = dist_vocab[instance.target]

    @staticmethod
    def add_transformed_distr(instances, dist_vocab):
        for instance in instances:
            instance.xs_transf_vec = dist_vocab[instance.source]
