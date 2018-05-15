import dynet as dy
import os
import random
import sys
from tqdm import tqdm

from morphology.models.seq2seq import evaluate


class Trainer(object):
    def __init__(self, pc, model, model_file, lr, decay_rate, searcher):
        self.pc = pc
        self.model = model
        self.model_file = model_file
        self.lr = lr
        self.decay_rate = decay_rate
        self.searcher = searcher

        self.optimizer = dy.AdamTrainer(pc, alpha=lr)


    def train(self, train, dev, epochs, batch_size):
        # First copy the train and dev so we can shuffle them
        train, dev = train[:], dev[:]
        try:
            best_loss = sys.maxsize
            lr = self.lr

            for t in tqdm(range(epochs), desc='Epochs'):
                train_loss = self.one_epoch(train, batch_size)
                dev_loss = evaluate.total_loss(self.model, dev)
                dev_preds = evaluate.generate(self.model, self.searcher, dev)
                dev_acc, dev_edit = evaluate.accuracy(dev, dev_preds)
                tqdm.write('Iteration: {}\tTrain Loss: {:.2f}\tDev Loss: {:.2f}\t'
                           'Dev Accuracy: {:.2f}\tDev Edit: {:.2f}'.format(
                            t, train_loss, dev_loss, dev_acc, dev_edit))
                if dev_loss < best_loss:
                    best_loss = dev_loss
                    os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
                    self.pc.save(self.model_file)
                    tqdm.write('Saving best model')
                else:
                    self.pc.populate(self.model_file)
                    lr = lr * self.decay_rate
                    self.optimizer.restart(lr)
                    tqdm.write('Reverting to previous checkpoint, lr = {}'.format(lr))

        except KeyboardInterrupt:
            print('Terminating training early')

        # Load the best model
        self.pc.populate(self.model_file)

    def one_epoch(self, train, batch_size):
        batch_loss = []
        total_loss = 0
        random.shuffle(train)
        for i, instance in enumerate(tqdm(train, desc='Training')):
            loss = evaluate.nll(self.model, instance)
            batch_loss.append(loss)

            if i % batch_size == 0:
                loss = dy.esum(batch_loss)
                total_loss += loss.value()

                loss.backward()
                self.optimizer.update()
                dy.renew_cg()
                batch_loss = []

        return total_loss
