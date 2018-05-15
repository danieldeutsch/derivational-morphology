class Searcher(object):
    def __init__(self):
        self.examples = 0
        self.states = 0

    def next_state(self):
        self.states += 1

    def next_example(self):
        self.examples += 1

    def reset(self):
        self.states = 0
        self.examples = 0

    def average_states_per_example(self):
        return self.states / self.examples
