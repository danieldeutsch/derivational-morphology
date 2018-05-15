import glob
import os
import sys
from collections import defaultdict

splits = ['Train', 'Dev', 'Test']
metrics = ['Accuracy', 'Average States'] #, 'Average Time']
methods = set()

expt_dir = sys.argv[1]
stats = defaultdict(list)
for path in glob.glob(os.path.join(expt_dir, '*/metrics.txt')):
    lines = open(path, 'r').read().splitlines()
    for line in lines[1:]:
        method, split, acc, states, time = line.split('\t')
        methods.add(method)
        stats['{}_{}_{}'.format(method, split, 'Accuracy')].append(float(acc))
        stats['{}_{}_{}'.format(method, split, 'Average States')].append(float(states))
        stats['{}_{}_{}'.format(method, split, 'Average Time')].append(float(time))

with open(os.path.join(expt_dir, 'summary.txt'), 'w') as out:
    for method in methods:
        for split in splits:
            for metric in metrics:
                values = stats['{}_{}_{}'.format(method, split, metric)]
                avg = sum(values) / len(values)
                out.write('{}\t{}\t{}\t{}\n'.format(method, split, metric, avg))
