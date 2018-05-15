import dynet as dy
import json
import numpy
import random
import sys
from tqdm import tqdm

_initializer = dy.UniformInitializer(0.1)
_zero_initializer = dy.ConstInitializer(0.0)


class Ensembler:
    '''
    Learns to choose from outputs of two independent systems
    with differing sources of information
    '''

    def __init__(self, irregularity_model_file, vocab, args):
        self.vocab = vocab
        self.epochs = args['epochs'] if 'epochs' in args else 10
        self.decay_rate = args['decay_rate'] if 'decay_rate' in args else .5
        self.lr = args['lr'] if 'lr' in args else 0.005
        self.batch_size = args['batch-size'] if 'batch-size' in args else 1
        self.choices = args['num-choices'] if 'num-choices' in args else 2
        self.model_file = irregularity_model_file

        self.pc = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.pc, alpha=self.lr)

    def define_params(self, observations):
        self.param_dict = {}
        self.known_transformations = set()
        for observation in observations:
            #transformation = observation.transformation
            transformation = 'lul'
            if transformation not in self.known_transformations:
                self.param_dict[(transformation, 'b')] = self.pc.add_parameters((self.choices), _zero_initializer)
                self.param_dict[(transformation, 'W')] = self.pc.add_parameters((self.choices, self.choices), _initializer)
                self.known_transformations.add(transformation)


    def train(self, train_seq, train_transf, dev_seq, dev_transf, train_instances, dev_instances):
        assert len(train_seq) == len(train_transf)
        assert len(dev_seq) == len(dev_transf)
        assert len(train_seq) == len(train_instances)
        assert len(dev_seq) == len(dev_instances)

        train_seq = [self.remove_header_instance(x) for x in train_seq]
        train_seq = [self.stringify_beam(x) for x in train_seq]
        dev_seq = [self.remove_header_instance(x) for x in dev_seq]
        dev_seq = [self.stringify_beam(x) for x in dev_seq]

        train_disagreement_set = list(filter(self.filter_to_disagreements,
            zip(train_seq, train_transf, train_instances)))
        dev_disagreement_set = list(filter(self.filter_to_disagreements,
            zip(dev_seq, dev_transf, dev_instances)))
        #print(len(train_disagreement_set))
        #print(len(dev_disagreement_set))

        train = [(self.features_of_topks(shyp, thyp, instance), instance) for shyp, thyp, instance in
                train_disagreement_set]
        dev = [(self.features_of_topks(shyp, thyp, instance), instance) for shyp, thyp, instance in
                dev_disagreement_set]

        #print('transf correct, train', len(list(filter(lambda x: x[0][1] == 1, train))))
        #print('transf correct, dev', len(list(filter(lambda x: x[0][1] == 1, dev))))

        self._train(train, dev)



    def _train(self, train, dev):
        best_loss = sys.maxsize
        #train = list(zip(train, train_instances))
        #dev = list(zip(dev, dev_instances))
        for t in tqdm(range(self.epochs), desc='[picker epoch]'):
            random.shuffle(train)
            train_loss = self.epoch(train)
            dev_loss = self.evaluate(dev)
            tqdm.write('Iteration: {}\tTrain Loss: {:.2f}\tDev Loss: {:.2f}'
                    .format(t, float(train_loss), float(dev_loss)))
            if dev_loss < best_loss:
                best_loss = dev_loss
                self.pc.save(self.model_file)
                tqdm.write('Saving best picker model')
            else:
                self.pc.populate(self.model_file)
                self.lr = self.lr * self.decay_rate
                self.trainer.restart(self.lr)
                tqdm.write('Reverting picker to previous checkpoint, lr = {}'
                        .format(self.lr))

    def epoch(self, observations):
        batch_loss = []
        total_loss = 0
        for i, (observation, instance) in enumerate(tqdm(observations, desc='[picker instance]')):
            _, loss = self.loss(observation, instance)
            batch_loss.append(loss)
            if i % self.batch_size == 0:
                loss = dy.esum(batch_loss)
                total_loss += loss.value()
                loss.backward()
                self.trainer.update()
                dy.renew_cg()
                batch_loss = []
        return total_loss

    def evaluate(self, observations):
        total_loss = 0
        for observation, instance in observations:
            _, loss = self.loss(observation, instance)
            total_loss += loss.value()
            dy.renew_cg()
        return total_loss

    def __call__(self, seq_beam, transf_beam, instance):
        seq_beam = self.remove_header_instance(seq_beam)
        observation = self.features_of_topks(seq_beam, transf_beam, None)
        transf_beam = self.integerize_beam(transf_beam)

        prediction_probs, _ = self.loss(observation, instance)
        choice = numpy.argmax(prediction_probs.value())

        chosen_hyp = [seq_beam, transf_beam][choice][0]

        return self.add_header_instance(chosen_hyp[1], [chosen_hyp])

    def stringify_beam(self, beam):
        newbeam = []
        for hyp in beam:
            score, word = hyp
            string_word = ''.join([self.vocab[x] for x in word[1:-1]])
            newbeam.append((score, string_word))
        return newbeam

    def integerize_beam(self, beam):
        newbeam = []
        for hyp in beam:
            score, word = hyp
            newbeam.append((score, [self.vocab['#']] + [self.vocab[x] for x in word] + [self.vocab['#']]))
        return newbeam

    def loss(self, observation, instance):
        #trans = instance.transformation
        #if trans not in self.known_transformations:
        #k    newtrans = list(self.param_dict.keys())[0][0] ### SUPER ARBITRARY
        #k    tqdm.write("WARNING: unknown transformtion picked for instance {}; using transformation {}".format(trans, newtrans))
        #k    trans = newtrans

        trans = 'lul'
        b = dy.parameter(self.param_dict[(trans, 'b')])
        W = dy.parameter(self.param_dict[(trans, 'W')])

        features, label = observation

        prediction = dy.softmax(dy.affine_transform([b, W, dy.inputVector(features)]))

        loss = -dy.log(dy.pick(prediction, label))

        return prediction, loss

    @staticmethod
    def remove_header_instance(beam):
        return beam[1]

    @staticmethod
    def add_header_instance(instance, beam):
        return [instance, beam]

    @staticmethod
    def features_of_topks(seq_beam, transf_beam, answer):
        features = [seq_beam[0][0], transf_beam[0][0]]
        if answer == None:
            return (features, 0)
        label = 0 if seq_beam[0][1] == answer.target else -1
        label = 1 if (label == -1 and transf_beam[0][1] == answer.target) else label
        if label == -1:
            print('label error for', answer, seq_beam, transf_beam)
        return (features, label)

    @staticmethod
    def filter_to_disagreements(args):
        seq_beam, transf_beam, answer = args
        if seq_beam[0][1] != transf_beam[0][1]:
            if seq_beam[0][1] == answer.target or transf_beam[0][1] == answer.target:
                #print('disagreement', seq_beam, transf_beam, answer)
                return True
        #print('agreement', seq_beam, transf_beam, answer)
        return False

    @staticmethod
    def load_transf_hyps(filepath, instances):
        beams = [json.loads(line.strip()) for line in open(filepath)]
        newbeam = []
        for item, instance in zip(beams, instances):
            newbeam.append(list(filter(
                lambda x: x[1].startswith(instance.source[:4]) and x[1] != instance.source and '_' not in x[1],
                item))+ [(0, "")])
        return newbeam
