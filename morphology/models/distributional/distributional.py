import dynet as dy
import numpy
import random
import sys
from tqdm import tqdm


class DistributionalTransformer:

    def __init__(self, vec_model, transformer_model_file, args):
        self.vec_model = vec_model
        self.epochs = args['epochs'] if 'epochs' in args else 2
        self.decay_rate = args['decay_rate'] if 'decay_rate' in args else .5
        self.lr = args['lr'] if 'lr' in args else 0.005
        self.batch_size = args['batch-size'] if 'batch-size' in args else 16
        self.vec_size = args['vec-size'] if 'vec-size' in args else 300
        self.model_file = transformer_model_file
        self.hidden_dim = 100

        self.pc = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.pc, alpha=self.lr)

    def define_params(self, instances):
        self.param_dict = {}
        self.known_transformations = set()
        for instance in instances:
            transformation = instance.transformation
            #transformation = 'lol'
            if transformation not in self.known_transformations:
                self.param_dict[(transformation, 'b1')] = self.pc.add_parameters((self.hidden_dim))
                self.param_dict[(transformation, 'W1')] = self.pc.add_parameters(
                        (self.hidden_dim, self.vec_size))
                self.param_dict[(transformation, 'b2')] = self.pc.add_parameters((self.vec_size))
                self.param_dict[(transformation, 'W2')] = self.pc.add_parameters(
                        (self.vec_size, self.hidden_dim))
                #self.param_dict[(transformation, 'b3')] = self.pc.add_parameters((self.vec_size))
                #self.param_dict[(transformation, 'W3')] = self.pc.add_parameters(
                #        (self.vec_size, self.hidden_dim))
                #self.param_dict[(transformation, 'b')] = self.pc.add_parameters((self.vec_size))
                #self.param_dict[(transformation, 'W')] = self.pc.add_parameters((self.vec_size, self.vec_size))
                self.known_transformations.add(transformation)
        print(self.known_transformations)
        print(self.param_dict)

    def train(self, train_instances, dev_instances):
        best_loss = sys.maxsize
        for t in tqdm(range(self.epochs), desc='[transf epoch]'):
            random.shuffle(train_instances)
            train_loss = self.epoch(train_instances)
            dev_loss = self.evaluate(dev_instances)
            tqdm.write('Iteration: {}\tTrain Loss: {:.2f}\tDev Loss: {:.2f}\t'
                    .format(t, train_loss, dev_loss))
            if dev_loss < best_loss:
                best_loss = dev_loss
                self.pc.save(self.model_file)
                tqdm.write('Saving best transformation model')
            else:
                self.pc.populate(self.model_file)
                self.lr = self.lr * self.decay_rate
                self.trainer.restart(self.lr)
                tqdm.write('Reverting transformer to previous checkpoint, lr = {}'
                        .format(self.lr))

    def epoch(self, instances):
        batch_loss = []
        total_loss = 0
        for i, instance in enumerate(tqdm(instances, desc='[transf instance]')):
            _, loss = self.loss(instance)
            batch_loss.append(loss)
            #if not numpy.any(instance.xs_distr_vec) or not numpy.any(instance.ys_distr_vec):
            #    continue
            if i % self.batch_size == 0:
                loss = dy.esum(batch_loss)
                total_loss += loss.value()
                loss.backward()
                self.trainer.update()
                dy.renew_cg()
                batch_loss = []
        return total_loss

    def evaluate(self, instances):
        total_loss = 0
        for instance in instances:
            if not numpy.any(instance.xs_distr_vec) or not numpy.any(instance.ys_distr_vec):
                #tqdm.write('Skipping instance {} because of empty vec'.format(instance))
                #print('Skipping instance {} because of empty vec'.format(instance))
                continue
            _, loss = self.loss(instance)
            total_loss += loss.value()
            dy.renew_cg()
        return total_loss

    def __call__(self, instance):
        #string_vec = self.vec_model[string]
        if not numpy.any(instance.xs_distr_vec):
            #tqdm.write('WARNING: empty string vec for instance: {}'.format(instance))
            pass
        prediction, _ = self.loss(instance)
        return prediction.value()

    def loss(self, instance):
        trans = instance.transformation
        #trans = 'lol'
        if trans not in self.known_transformations:
            newtrans = list(self.param_dict.keys())[0][0] ### SUPER ARBITRARY
            tqdm.write("WARNING: unknown transformtion picked for instance {}; using transformation {}".format(trans, newtrans))
            trans = newtrans
        b1 = dy.parameter(self.param_dict[(trans, 'b1')])
        W1 = dy.parameter(self.param_dict[(trans, 'W1')])
        b2 = dy.parameter(self.param_dict[(trans, 'b2')])
        W2 = dy.parameter(self.param_dict[(trans, 'W2')])
        #b3 = dy.parameter(self.param_dict[(trans, 'b3')])
        #W3 = dy.parameter(self.param_dict[(trans, 'W3')])

        #b = dy.parameter(self.param_dict[(trans, 'b')])
        #W = dy.parameter(self.param_dict[(trans, 'W')])

        x = dy.inputVector(instance.xs_distr_vec)
        y = dy.inputVector(instance.ys_distr_vec)

        #prediction = dy.affine_transform([b, W, x])
        prediction = dy.affine_transform(
            [b2, W2, dy.tanh(dy.affine_transform([b1, W1, x])) ] )
        #prediction = dy.affine_transform(
        #        [b3, W3, dy.tanh(dy.affine_transform(
        #        [b2, W2, dy.tanh(dy.affine_transform([b1, W1, x])) ] ))])

        loss = dy.squared_distance(prediction, y)

        return prediction, loss
