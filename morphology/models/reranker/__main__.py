import argparse
import numpy as np
import random
import yaml

from morphology import Instance, util
from morphology.models.reranker import MlpReranker
from morphology.models.seq2seq import PrefixTree, evaluate
from morphology.vocab import Vocab


def main(args):
    params = yaml.load(open(args.config_file, 'r'))

    reranking_features = params['reranker-params']['reranking-features']
    ptree_type = params['decoder-params']['ptree-type']
    min_counts = params['decoder-params']['min-counts']

    train = Instance.load(args.train_src, args.train_trg)
    valid = Instance.load(args.valid_src, args.valid_trg)
    test = Instance.load(args.test_src, args.test_trg)

    vocab = Vocab.load(args.vocab_file)

    Instance.encode(train, vocab)
    Instance.encode(valid, vocab)
    Instance.encode(test, vocab)

    seq2seq_train_preds = util.load_predictions(vocab, args.train_seq2seq_preds)
    seq2seq_valid_preds = util.load_predictions(vocab, args.valid_seq2seq_preds)
    seq2seq_test_preds = util.load_predictions(vocab, args.test_seq2seq_preds)

    # Build the prefix tree for the dictionary constraint
    if ptree_type == 'prefix':
        ptree = PrefixTree(args.unigram_counts, vocab, min_counts)
    elif ptree_type == 'suffix':
        ptree = PrefixTree(args.unigram_counts, vocab, min_counts, train)
    else:
        raise Exception('Unknown ptree type: ' + ptree_type)

    reranker = MlpReranker(reranking_features, args.model_file, ptree, vocab, None)
    reranker.train(seq2seq_train_preds, train, seq2seq_valid_preds, valid)

    train_preds = [reranker(beam, inst) for beam, inst in zip(seq2seq_train_preds, train)]
    valid_preds = [reranker(beam, inst) for beam, inst in zip(seq2seq_valid_preds, valid)]
    test_preds = [reranker(beam, inst) for beam, inst in zip(seq2seq_test_preds, test)]

    util.save_predictions(train_preds, vocab, args.train_output)
    util.save_predictions(valid_preds, vocab, args.valid_output)
    util.save_predictions(test_preds, vocab, args.test_output)

    train_metrics = util.calculate_metrics(train, train_preds)
    valid_metrics = util.calculate_metrics(valid, valid_preds)
    test_metrics = util.calculate_metrics(test, test_preds)

    util.save_metrics(train_metrics, 'reranker', 'train', args.metrics_file)
    util.save_metrics(valid_metrics, 'reranker', 'valid', args.metrics_file)
    util.save_metrics(test_metrics, 'reranker', 'test', args.metrics_file)

    if args.held_out_srcs is not None:
        for src, trg, seq2seq_preds_json, output in zip(args.held_out_srcs,
                                                        args.held_out_trgs,
                                                        args.held_out_seq2seq_preds,
                                                        args.held_out_outputs):
            held_out = Instance.load(src, trg)
            seq2seq_preds = util.load_predictions(vocab, seq2seq_preds_json)
            Instance.encode(held_out, vocab)
            preds = [reranker(beam, inst) for beam, inst in zip(seq2seq_preds, held_out)]
            util.save_predictions(preds, vocab, output)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    # Inputs
    argp.add_argument('--config-file', required=True)
    argp.add_argument('--train-src', required=True)
    argp.add_argument('--train-trg', required=True)
    argp.add_argument('--train-seq2seq-preds', required=True)
    argp.add_argument('--valid-src', required=True)
    argp.add_argument('--valid-trg', required=True)
    argp.add_argument('--valid-seq2seq-preds', required=True)
    argp.add_argument('--test-src', required=True)
    argp.add_argument('--test-trg', required=True)
    argp.add_argument('--test-seq2seq-preds', required=True)
    argp.add_argument('--held-out-srcs', nargs='+')
    argp.add_argument('--held-out-trgs', nargs='+')
    argp.add_argument('--held-out-seq2seq-preds', nargs='+')
    argp.add_argument('--unigram-counts', required=True)
    argp.add_argument('--vocab-file', required=True)
    # Outputs
    argp.add_argument('--model-file', required=True)
    argp.add_argument('--train-output', required=True)
    argp.add_argument('--valid-output', required=True)
    argp.add_argument('--test-output', required=True)
    argp.add_argument('--metrics-file', required=True)
    argp.add_argument('--held-out-outputs', nargs='+')
    # Optional parameters
    argp.add_argument('--dynet-seed')
    args = argp.parse_args()

    # The DyNet seed is set via the command line argument
    if args.dynet_seed is not None:
        random.seed(int(args.dynet_seed))
        np.random.seed(int(args.dynet_seed))

    main(args)
