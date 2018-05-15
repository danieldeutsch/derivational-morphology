import argparse
import json
import numpy as np
import random
import yaml
from tqdm import tqdm

from morphology import Instance, util
from morphology.models.distributional import DistributionalTransformer
from morphology.vocab import BOS, DistributionalVocab, Vocab


def select_hyps(vecs, model):
    hyps = []
    for vec in tqdm(vecs, desc='Generating hypotheses'):
        similars = model.similar_by_vector(vec, topn=60, restrict_vocab=None)
        similars = [(x[1], x[0]) for x in similars]
        hyps.append(similars)
    return hyps


def filter_hyps(hyps, instances):
    newbeam = []
    for item, instance in zip(hyps, instances):
        newbeam.append(list(filter(
            lambda x: x[1].startswith(instance.source[:4]) and x[1] != instance.source and '_' not in x[1],
            item)) + [(0, "")])
    return newbeam


def encode_hyps(hyps, vocab):
    assert BOS in vocab
    bos = vocab[BOS]

    encoded = []
    for beam in hyps:
        pred = beam[0][1]
        pred = [bos] + [vocab[x] for x in pred] + [bos]
        encoded_beam = []
        for score, value in beam:
            value = [bos] + [vocab[x] for x in value] + [bos]
            encoded_beam.append((score, value))
        encoded.append((pred, encoded_beam))
    return encoded


def write_hyps(hyps, output_path):
    with open(output_path, 'w') as fout:
        for similars in tqdm(hyps, desc='Writing hypotheses'):
            fout.write(json.dumps(similars) + '\n')


def main(args):
    params = yaml.load(open(args.config_file, 'r'))
    dist_args = params['dist-params']['args']

    train = Instance.load(args.train_src, args.train_trg)
    valid = Instance.load(args.valid_src, args.valid_trg)
    test = Instance.load(args.test_src, args.test_trg)

    vocab = Vocab.load(args.vocab_file)
    Instance.encode(train, vocab)
    Instance.encode(valid, vocab)
    Instance.encode(test, vocab)

    dvocab = DistributionalVocab(args.word_vectors)
    Instance.add_distr(train, dvocab)
    Instance.add_distr(valid, dvocab)
    Instance.add_distr(test, dvocab)

    transformer = DistributionalTransformer(dvocab, args.model_file, dist_args)
    transformer.define_params(train)
    transformer.train(train, valid)

    train_transformed_vecs = np.array([transformer(x) for x in train])
    valid_transformed_vecs = np.array([transformer(x) for x in valid])
    test_transformed_vecs = np.array([transformer(x) for x in test])

    print('Saving transformed xs vectors (into yhat vectors)')
    np.savetxt(args.train_vectors, train_transformed_vecs)
    np.savetxt(args.valid_vectors, valid_transformed_vecs)
    np.savetxt(args.test_vectors, test_transformed_vecs)

    print('Writing hypotheses')
    train_hyps = select_hyps(train_transformed_vecs, dvocab.vector_model)
    valid_hyps = select_hyps(valid_transformed_vecs, dvocab.vector_model)
    test_hyps = select_hyps(test_transformed_vecs, dvocab.vector_model)

    write_hyps(train_hyps, args.train_hypotheses)
    write_hyps(valid_hyps, args.valid_hypotheses)
    write_hyps(test_hyps, args.test_hypotheses)

    train_hyps = filter_hyps(train_hyps, train)
    valid_hyps = filter_hyps(valid_hyps, valid)
    test_hyps = filter_hyps(test_hyps, test)

    train_encoded_hyps = encode_hyps(train_hyps, vocab)
    valid_encoded_hyps = encode_hyps(valid_hyps, vocab)
    test_encoded_hyps = encode_hyps(test_hyps, vocab)

    train_metrics = util.calculate_metrics(train, train_encoded_hyps)
    valid_metrics = util.calculate_metrics(valid, valid_encoded_hyps)
    test_metrics = util.calculate_metrics(test, test_encoded_hyps)

    util.save_metrics(train_metrics, 'dist', 'train', args.metrics_file)
    util.save_metrics(valid_metrics, 'dist', 'valid', args.metrics_file)
    util.save_metrics(test_metrics, 'dist', 'test', args.metrics_file)

    if args.held_out_srcs is not None:
        for src, trg, vectors, hypos in zip(args.held_out_srcs,
                                            args.held_out_trgs,
                                            args.held_out_vectors,
                                            args.held_out_hypotheses):
            held_out = Instance.load(src, trg)
            Instance.add_distr(held_out, dvocab)
            vecs = np.array([transformer(x) for x in held_out])
            hyps = select_hyps(vecs, dvocab.vector_model)
            np.savetxt(vectors, vecs)
            write_hyps(hyps, hypos)


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
    argp.add_argument('--word-vectors', required=True)
    argp.add_argument('--vocab-file', required=True)
    # Outputs
    argp.add_argument('--model-file', required=True)
    argp.add_argument('--train-vectors', required=True)
    argp.add_argument('--train-hypotheses', required=True)
    argp.add_argument('--valid-vectors', required=True)
    argp.add_argument('--valid-hypotheses', required=True)
    argp.add_argument('--test-vectors', required=True)
    argp.add_argument('--test-hypotheses', required=True)
    argp.add_argument('--metrics-file', required=True)
    argp.add_argument('--held-out-vectors', nargs='+')
    argp.add_argument('--held-out-hypotheses', nargs='+')
    # Optional parameters
    argp.add_argument('--dynet-seed')
    args = argp.parse_args()

    # The DyNet seed is set via the command line argument
    if args.dynet_seed is not None:
        random.seed(int(args.dynet_seed))
        np.random.seed(int(args.dynet_seed))

    main(args)
