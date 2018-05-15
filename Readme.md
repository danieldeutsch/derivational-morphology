## Setup
To clone the repository with the dataset, shard the training data, and download the `unigram-counts.txt` data file, run
```
sh setup.sh
```
The `data/unigram-counts.txt` file is stored [here](https://www.seas.upenn.edu/~ddeutsch/acl2018/unigram-counts.txt), but can be optionally reproduced with the following command
```
sh scripts/ngram_counts.sh download data/unigram-counts.txt
```

The distributional model uses pretrained word vectors from https://code.google.com/archive/p/word2vec/. Please download the word vectors from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) to `data/GoogleNews-vectors-negative300.bin.gz`.

## Reproducing experiments
Our experiments are built on the Sun Grid Engine and use the `qsub` command to run several random seed restarts simultaneously.

To reproduce the accuracy and edit-distance results
```
sh scripts/run-experiment.sh unconstrained experiments/default.yaml
sh scripts/run-experiment.sh constrained experiments/constrained.yaml

python scripts/summarize_results.py unconstrained unconstrained-metrics.txt
python scripts/summarize_results.py constrained constrained-metrics.txt
python scripts/make_acc_tables.py unconstrained-metrics.txt constrained-metrics.txt
```

To reproduce the search results table
```
sh scripts/run-all-search-experiments.sh search experiments/search.yaml

python scripts/summarize_search_results.py search
python scripts/make_search_table.py search/summary.txt
```

To run the training for an individual model instead of the entire set of 30 random restarts, run one of the following scripts
```
scripts/train-dist.sh
scripts/train-seq2seq.sh
scripts/train-seq2seq-reranker.sh
scripts/train-dist-seq2seq.sh
scripts/train-dist-seq2seq-reranker.sh
```
Each script takes 3 arguments: the yaml config file (e.g. see `experiments/default.yaml` or `experiments/constrained.yaml`), an output directory, and a random seed.
