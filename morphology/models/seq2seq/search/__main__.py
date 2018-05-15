import argparse
import dynet as dy
import numpy as np
import random
import time
import yaml

from morphology import Instance
from morphology.models.seq2seq import (
    PrefixTree, Seq2SeqModel, Trainer, evaluate)
from morphology.models.seq2seq.search import (
    BeamSearch, ConstraintShortestPathSearch,
    ConstraintGreedySearch, ConstraintBeamSearch,
    GreedySearch, ShortestPathSearch,
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

    greedy = GreedySearch(bos, bos)
    beam = BeamSearch(beam_size, bos, bos)
    shortest = ShortestPathSearch(bos, bos)
    shortest_k = TopKShortestPathSearch(beam_size, bos, bos)

    constraint_greedy = ConstraintGreedySearch(bos, bos, ptree, vocab)
    constraint_beam = ConstraintBeamSearch(beam_size, bos, bos, ptree, vocab)
    constraint_shortest = ConstraintShortestPathSearch(bos, bos, ptree, vocab)
    constraint_shortest_k = ConstraintTopKShortestPathSearch(beam_size, bos, bos, ptree, vocab)

    # Just use greedy search during training since we don't care about
    # the accuracy
    trainer = Trainer(pc, model, args.model_file, lr, decay_rate, greedy)
    trainer.train(train, valid, epochs, batch_size)

    searchers = [greedy, beam, shortest, shortest_k, constraint_greedy, constraint_beam, constraint_shortest, constraint_shortest_k]
    names = ['greedy', 'beam', 'shortest', 'shortest-k', 'constraint-greedy', 'constraint-beam', 'constraint-shortest', 'constraint-shortest-k']

    train_accs, dev_accs, test_accs = [], [], []
    train_states, dev_states, test_states = [], [], []
    train_times, dev_times, test_times = [], [], []

    for searcher, name in zip(searchers, names):
        searcher.reset()
        start = time.time()
        train_preds = evaluate.generate(model, searcher, train)
        end = time.time()
        train_times.append((end - start) / len(train))
        train_states.append(searcher.average_states_per_example())

        searcher.reset()
        start = time.time()
        dev_preds = evaluate.generate(model, searcher, valid)
        end = time.time()
        dev_times.append((end - start) / len(valid))
        dev_states.append(searcher.average_states_per_example())

        searcher.reset()
        start = time.time()
        test_preds = evaluate.generate(model, searcher, test)
        end = time.time()
        test_times.append((end - start) / len(test))
        test_states.append(searcher.average_states_per_example())

        train_acc, train_edit = evaluate.accuracy(train, train_preds)
        dev_acc, dev_edit = evaluate.accuracy(valid, dev_preds)
        test_acc, test_edit = evaluate.accuracy(test, test_preds)

        train_accs.append(train_acc)
        dev_accs.append(dev_acc)
        test_accs.append(test_acc)

    with open(args.metrics_file, 'w') as out:
        out.write('Searcher\tSplit\tAccuracy\tAverage States\tAverage Time\n')
        for i in range(len(names)):
            out.write('{}\tTrain\t{}\t{}\t{}\n'.format(names[i], train_accs[i], train_states[i], train_times[i]))
            out.write('{}\tDev\t{}\t{}\t{}\n'.format(names[i], dev_accs[i], dev_states[i], dev_times[i]))
            out.write('{}\tTest\t{}\t{}\t{}\n'.format(names[i], test_accs[i], test_states[i], test_times[i]))


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
    argp.add_argument('--unigram-counts', required=True)
    argp.add_argument('--vocab-file', required=True)
    # Outputs
    argp.add_argument('--model-file', required=True)
    argp.add_argument('--metrics-file', required=True)
    # Optional parameters
    argp.add_argument('--dynet-seed')
    args = argp.parse_args()

    # The DyNet seed is set via the command line argument
    if args.dynet_seed is not None:
        random.seed(int(args.dynet_seed))
        np.random.seed(int(args.dynet_seed))

    main(args)
