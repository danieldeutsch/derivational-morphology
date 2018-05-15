import argparse
import numpy as np
import random
import yaml

from morphology import Instance, util
from morphology.models.ensemble import Ensembler
from morphology.models.seq2seq import evaluate
from morphology.vocab import Vocab


def main(args):
    params = yaml.load(open(args.config_file, 'r'))
    ensembler_args = params['ensembler-params']['ensembler-args']

    train = Instance.load(args.train_src, args.train_trg)
    valid = Instance.load(args.valid_src, args.valid_trg)
    test = Instance.load(args.test_src, args.test_trg)

    vocab = Vocab.load(args.vocab_file)

    Instance.encode(train, vocab)
    Instance.encode(valid, vocab)
    Instance.encode(test, vocab)

    ensembler = Ensembler(args.model_file, vocab, ensembler_args)
    ensembler.define_params(train)

    seq2seq_train_preds = util.load_predictions(vocab, args.train_seq2seq_preds)
    seq2seq_valid_preds = util.load_predictions(vocab, args.valid_seq2seq_preds)
    seq2seq_test_preds = util.load_predictions(vocab, args.test_seq2seq_preds)

    dist_train_hypos = Ensembler.load_transf_hyps(args.train_dist_hypos, train)
    dist_valid_hypos = Ensembler.load_transf_hyps(args.valid_dist_hypos, valid)
    dist_test_hypos = Ensembler.load_transf_hyps(args.test_dist_hypos, test)

    ensembler.train(seq2seq_train_preds, dist_train_hypos, seq2seq_valid_preds,
                    dist_valid_hypos, train, valid)

    train_preds = [ensembler(seq_beam, transf_beam, instance) for seq_beam, transf_beam, instance in zip(seq2seq_train_preds, dist_train_hypos, train)]
    valid_preds = [ensembler(seq_beam, transf_beam, instance) for seq_beam, transf_beam, instance in zip(seq2seq_valid_preds, dist_valid_hypos, valid)]
    test_preds = [ensembler(seq_beam, transf_beam, instance) for seq_beam, transf_beam, instance in zip(seq2seq_test_preds, dist_test_hypos, test)]

    util.save_predictions(train_preds, vocab, args.train_output)
    util.save_predictions(valid_preds, vocab, args.valid_output)
    util.save_predictions(test_preds, vocab, args.test_output)

    train_metrics = util.calculate_metrics(train, train_preds)
    valid_metrics = util.calculate_metrics(valid, valid_preds)
    test_metrics = util.calculate_metrics(test, test_preds)

    util.save_metrics(train_metrics, 'ensemble', 'train', args.metrics_file)
    util.save_metrics(valid_metrics, 'ensemble', 'valid', args.metrics_file)
    util.save_metrics(test_metrics, 'ensemble', 'test', args.metrics_file)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    # Inputs
    argp.add_argument('--config-file', required=True)
    argp.add_argument('--train-src', required=True)
    argp.add_argument('--train-trg', required=True)
    argp.add_argument('--train-seq2seq-preds', required=True)
    argp.add_argument('--train-dist-hypos', required=True)
    argp.add_argument('--valid-src', required=True)
    argp.add_argument('--valid-trg', required=True)
    argp.add_argument('--valid-seq2seq-preds', required=True)
    argp.add_argument('--valid-dist-hypos', required=True)
    argp.add_argument('--test-src', required=True)
    argp.add_argument('--test-trg', required=True)
    argp.add_argument('--test-seq2seq-preds', required=True)
    argp.add_argument('--test-dist-hypos', required=True)
    argp.add_argument('--vocab-file', required=True)
    # Outputs
    argp.add_argument('--model-file', required=True)
    argp.add_argument('--train-output', required=True)
    argp.add_argument('--valid-output', required=True)
    argp.add_argument('--test-output', required=True)
    argp.add_argument('--metrics-file', required=True)
    # Optional parameters
    argp.add_argument('--dynet-seed')
    args = argp.parse_args()

    # The DyNet seed is set via the command line argument
    if args.dynet_seed is not None:
        random.seed(int(args.dynet_seed))
        np.random.seed(int(args.dynet_seed))

    main(args)
