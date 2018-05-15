import argparse
import dynet as dy
import numpy as np
import random
import yaml

from morphology import Instance, util
from morphology.models.seq2seq import (
    PrefixTree, Seq2SeqModel, Trainer, evaluate)
from morphology.models.seq2seq.search import (
    BeamSearch, ConstraintShortestPathSearch,
    ConstraintApproximateShortestPathSearch, ConstraintBeamSearch,
    GreedySearch, ShortestPathSearch, ApproximateShortestPathSearch,
    TopKShortestPathSearch, ConstraintTopKShortestPathSearch)
from morphology.vocab import BOS, Vocab


def main(args):
    params = yaml.load(open(args.config_file, 'r'))

    # Model parameters
    attention = params['model-params']['attention']
    batch_size = params['model-params']['batch-size']
    embed_dim = params['model-params']['embed-dim']
    epochs = params['model-params']['epochs']
    hidden_dim = params['model-params']['hidden-dim']
    num_layers = params['model-params']['num-layers']

    # Optimization parameters
    lr = params['optimization-params']['lr']
    decay_rate = params['optimization-params']['decay-rate']

    # Decoder parameters
    search_type = params['decoder-params']['search-type']
    ptree_type = params['decoder-params']['ptree-type']
    beam_size = params['decoder-params']['beam-size']
    min_counts = params['decoder-params']['min-counts']

    # Load the training data
    train = Instance.load(args.train_src, args.train_trg)
    valid = Instance.load(args.valid_src, args.valid_trg)
    test = Instance.load(args.test_src, args.test_trg)

    vocab = Vocab.load(args.vocab_file)
    assert BOS in vocab
    bos = vocab[BOS]

    Instance.encode(train, vocab)
    Instance.encode(valid, vocab)
    Instance.encode(test, vocab)

    # Build the model
    pc = dy.ParameterCollection()
    V = len(vocab)
    model = Seq2SeqModel(pc, V, embed_dim, hidden_dim, num_layers=num_layers,
                         attention=attention)

    # Build the prefix tree for the dictionary constraint
    if ptree_type == 'prefix':
        ptree = PrefixTree(args.unigram_counts, vocab, min_counts)
    elif ptree_type == 'suffix':
        ptree = PrefixTree(args.unigram_counts, vocab, min_counts, train)
    elif ptree_type == 'none':
        print("Not loading dictionary counts for ptree.")
    else:
        raise Exception('Unknown ptree type: ' + ptree_type)

    # Select the decoding search strategy
    if search_type == 'greedy':
        searcher = GreedySearch(bos, bos)
    elif search_type == 'beam':
        searcher = BeamSearch(beam_size, bos, bos)
    elif search_type == 'shortest':
        searcher = ShortestPathSearch(bos, bos)
    elif search_type == 'approx':
        searcher = ApproximateShortestPathSearch(beam_size, bos, bos)
    elif search_type == 'constraint-beam':
        searcher = ConstraintBeamSearch(beam_size, bos, bos, ptree, vocab)
    elif search_type == 'constraint-shortest':
        searcher = ConstraintShortestPathSearch(bos, bos, ptree, vocab)
    elif search_type == 'constraint-approx':
        searcher = ConstraintApproximateShortestPathSearch(beam_size, bos, bos,
                                                           ptree, vocab)
    elif search_type == 'shortest-k':
        searcher = TopKShortestPathSearch(beam_size, bos, bos)
    elif search_type == 'constraint-shortest-k':
        searcher = ConstraintTopKShortestPathSearch(beam_size, bos, bos, ptree,
                                                    vocab)

    trainer = Trainer(pc, model, args.model_file, lr, decay_rate, searcher)
    trainer.train(train, valid, epochs, batch_size)

    train_preds = evaluate.generate(model, searcher, train)
    valid_preds = evaluate.generate(model, searcher, valid)
    test_preds = evaluate.generate(model, searcher, test)

    util.save_predictions(train_preds, vocab, args.train_output)
    util.save_predictions(valid_preds, vocab, args.valid_output)
    util.save_predictions(test_preds, vocab, args.test_output)

    train_metrics = util.calculate_metrics(train, train_preds)
    valid_metrics = util.calculate_metrics(valid, valid_preds)
    test_metrics = util.calculate_metrics(test, test_preds)

    util.save_metrics(train_metrics, 'seq2seq', 'train', args.metrics_file)
    util.save_metrics(valid_metrics, 'seq2seq', 'valid', args.metrics_file)
    util.save_metrics(test_metrics, 'seq2seq', 'test', args.metrics_file)

    if args.held_out_srcs is not None:
        for src, trg, output in zip(args.held_out_srcs, args.held_out_trgs,
                                    args.held_out_outputs):
            held_out = Instance.load(src, trg)
            Instance.encode(held_out, vocab)
            preds = evaluate.generate(model, searcher, held_out)
            util.save_predictions(preds, vocab, output)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    # Inputs
    argp.add_argument('--config-file', required=True)
    argp.add_argument('--train-src', required=True)
    argp.add_argument('--train-trg', required=True)
    argp.add_argument('--valid-src', required=True)
    argp.add_argument('--valid-trg', required=True)
    argp.add_argument('--test-src', required=True)
    argp.add_argument('--test-trg', required=True)
    argp.add_argument('--held-out-srcs', nargs='+')
    argp.add_argument('--held-out-trgs', nargs='+')
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
