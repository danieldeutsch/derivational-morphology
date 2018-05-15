import argparse
import glob
import os
from collections import defaultdict


def main(args):
    with open(args.output_file, 'w') as out:
        for model_path in glob.glob(os.path.join(args.experiment_dir, '*')):
            model = os.path.basename(model_path)
            num_runs = 0
            metrics = defaultdict(float)
            for run_path in glob.glob(os.path.join(model_path, '*')):
                with open(os.path.join(run_path, 'metrics.txt')) as f:
                    for line in f:
                        component, split, metric, value = line.strip().split('\t')
                        metrics[f'{component}\t{split}\t{metric}'] += float(value)
                num_runs += 1

            for metric in sorted(metrics.keys()):
                value = metrics[metric] / num_runs
                out.write(f'{model}\t{metric}\t{value}\n')


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('experiment_dir')
    argp.add_argument('output_file')
    args = argp.parse_args()
    main(args)
