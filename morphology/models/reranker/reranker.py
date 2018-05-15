import dynet as dy
import math
import random
import sys
import numpy
from tqdm import tqdm
from scipy.spatial.distance import cosine

_initializer = dy.UniformInitializer(0.1)
_zero_initializer = dy.ConstInitializer(0.0)


MAX_SCORE = 100


class MlpReranker:

    def __init__(self, reranking_features, reranker_model_file,
            ptree, vocab, dvocab, feature_function=None, reranker_config={}):
        if len(reranking_features) < 2:
            raise Exception("Must have at least 2 reranking features.")

        self.pc = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.pc, alpha=.005)
        self.ptree = ptree
        self.vocab = vocab
        self.features = reranking_features
        self.dvocab = dvocab

        hidden_dim = len(reranking_features) + 1
        input_dim = len(reranking_features)

        self.b1 = self.pc.add_parameters((hidden_dim), _zero_initializer)
        self.W1 = self.pc.add_parameters((hidden_dim, input_dim), _initializer)
        self.b2 = self.pc.add_parameters((1), _zero_initializer)
        self.W2 = self.pc.add_parameters((1, hidden_dim), _initializer)

        self.EPOCHS = 5
        self.decay_rate = .5
        self.lr = 0.005
        self.batch_size = 8
        self.model_file = reranker_model_file

        self.feature_function = feature_function if feature_function else self.get_features_of_hyp

    def flatten(self, beams, data):
        for beam, observation in zip(beams, data):
            for beam_hyp in beam[1]:
                yield observation, beam_hyp

    def train(self, train_beam, train_data, dev_beam, dev_data):
        assert len(train_beam) == len(train_data)
        assert len(dev_beam) == len(dev_data)

        new_dev_beam = dev_beam

        train_obs = [self.get_features_of_hyp(instance, *beam_entry)
                for instance, beam_entry in self.flatten(train_beam, train_data)]
        dev_obs = [self.get_features_of_hyp(instance, *beam_entry)
                for instance, beam_entry in self.flatten(new_dev_beam, dev_data)]

        best_loss = sys.maxsize

        for t in tqdm(range(self.EPOCHS), desc='Epochs'):
            random.shuffle(train_obs)
            train_loss = self.epoch(train_obs)
            dev_loss = self.evaluate(dev_obs)
            dev_acc = self.accuracy(dev_beam, dev_data)
            tqdm.write('Iteration: {}\tTrain Loss: {:.2f}\tDev Loss: {:.2f}\t'
                       'Dev Accuracy: {:.2f}'.format(t, train_loss, dev_loss, dev_acc))
            if dev_loss < best_loss:
                best_loss = dev_loss
                self.pc.save(self.model_file)
                tqdm.write('Saving best reranking model')
            else:
                self.pc.populate(self.model_file)
                self.lr = self.lr * self.decay_rate
                self.trainer.restart(self.lr)
                tqdm.write('Reverting reranker to previous checkpoint, lr = {}'
                        .format(self.lr))

    def epoch(self, obs):
        batch_loss = []
        total_loss = 0
        for i, (features, label) in enumerate(tqdm(obs, desc='Reranker training')):
            _, loss = self.loss(features, label)
            batch_loss.append(loss)

            if i % self.batch_size == 0:
                loss = dy.esum(batch_loss)
                total_loss += loss.value()

                loss.backward()
                self.trainer.update()
                dy.renew_cg()
                batch_loss = []

        return total_loss

    def evaluate(self, obs):
        losses = []
        for features, label in obs:
            _, loss = self.loss(features, label)
            losses.append(loss.value())
            dy.renew_cg()
        return sum(losses)

    def symbolize(self, beam):
        for _, hyps in beam:
            yield [(score, ''.join([self.vocab[x] for x in ints])) for score, ints in hyps]

    def __call__(self, beam, observation):
        beam_feats = [self.get_features_of_hyp(instance, *beam_entry)[0]
                for instance, beam_entry in self.flatten([beam], [observation])]
        reranked = list(sorted([(self.score(feats).value(), ints)
            for feats, (_, ints) in zip(beam_feats, beam[1])], key = lambda x:-x[0]))
        return reranked[0][1], reranked

    def score(self, features):
        prediction, _ = self.loss(features, 0)
        return prediction

    def loss(self, features, y):

        b1 = dy.parameter(self.b1)
        W1 = dy.parameter(self.W1)
        b2 = dy.parameter(self.b2)
        W2 = dy.parameter(self.W2)

        x = dy.inputVector(features)

        prediction = dy.tanh(dy.affine_transform(
            [b2, W2, dy.tanh(dy.affine_transform([b1, W1, x])) ] ))

        loss = dy.square(prediction - y)

        return prediction, loss

    def accuracy(self, beams, instances):
        total = 0
        correct = 0
        for beam, instance in zip(beams, instances):
            reranked_beam = self(beam, instance)
            total += 1
            if reranked_beam[0][1] == instance.ys:
                correct += 1
        return correct/total

    def get_features_of_hyp(self, instance, score, prediction):
        predicted_word = prediction[1:-1]
        length_ratio = (len(instance.xs)-3)/max(len(predicted_word), 1)
        count = self.ptree.count(''.join([self.vocab[x] for x in predicted_word]))
        log_count = math.log(count) if count > 0 else 0
        features = []

        label = 1 if instance.ys == prediction else -1

        if 'model-score' in self.features:
            features.append(min(score, MAX_SCORE))
        if 'length-ratio' in self.features:
            features.append(length_ratio)
        if 'log-count' in self.features:
            features.append(log_count)
        if 'distr-sim' in self.features:
            prediction_vector = self.dvocab[''.join([self.vocab[x] for x in predicted_word])]
            cosine_value = cosine(prediction_vector, instance.xs_distr_vec)
            if numpy.isnan(cosine_value):
                cosine_value = 0
            features.append(cosine_value)
        if 'transf-sim' in self.features:
            prediction_vector = self.dvocab[''.join([self.vocab[x] for x in predicted_word])]
            cosine_value = cosine(prediction_vector, instance.xs_transf_vec)
            if numpy.isnan(cosine_value):
                cosine_value = 0
            features.append(cosine_value)
        return features, label
