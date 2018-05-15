#!/bin/csh
#$ -cwd
#$ -l mem=16G
config_file=$1
output_dir=$2
seed=${3:-4}

echo "Start - `date`"
>&2 echo "Start - `date`"

mkdir -p ${output_dir}

python -m morphology.models.distributional \
  --config-file ${config_file} \
  --train-src data/train.85-7-8.85.src \
  --train-trg data/train.85-7-8.85.trg \
  --valid-src data/dev.src \
  --valid-trg data/dev.trg \
  --test-src data/test.src \
  --test-trg data/test.trg \
  --word-vectors data/GoogleNews-vectors-negative300.bin.gz \
  --vocab-file data/vocab.txt \
  --model-file ${output_dir}/dist.pt \
  --train-vectors ${output_dir}/train.85-7-8.85.vecs \
  --train-hypotheses ${output_dir}/train.85-7-8.85.distributional.out \
  --valid-vectors ${output_dir}/valid.vecs \
  --valid-hypotheses ${output_dir}/valid.distributional.out \
  --test-vectors ${output_dir}/test.vecs \
  --test-hypotheses ${output_dir}/test.distributional.out \
  --metrics-file ${output_dir}/metrics.txt \
  --held-out-srcs data/train.85-7-8.7.src data/train.85-7-8.8.src \
  --held-out-trgs data/train.85-7-8.7.trg data/train.85-7-8.8.trg \
  --held-out-vectors ${output_dir}/train.85-7-8.7.vecs ${output_dir}/train.85-7-8.8.vecs \
  --held-out-hypotheses ${output_dir}/train.85-7-8.7.distributional.out ${output_dir}/train.85-7-8.8.distributional.out \
  --dynet-seed ${seed}

python -m morphology.models.seq2seq \
  --config-file ${config_file} \
  --train-src data/train.85-7-8.85.src \
  --train-trg data/train.85-7-8.85.trg \
  --valid-src data/dev.src \
  --valid-trg data/dev.trg \
  --test-src data/test.src \
  --test-trg data/test.trg \
  --vocab-file data/vocab.txt \
  --unigram-counts data/unigram-counts.txt \
  --model-file ${output_dir}/seq2seq.pt \
  --train-output ${output_dir}/train.85-7-8.85.seq2seq.json \
  --valid-output ${output_dir}/valid.seq2seq.json \
  --test-output ${output_dir}/test.seq2seq.json \
  --metrics-file ${output_dir}/metrics.txt \
  --held-out-srcs data/train.85-7-8.7.src data/train.85-7-8.8.src \
  --held-out-trgs data/train.85-7-8.7.trg data/train.85-7-8.8.trg \
  --held-out-outputs ${output_dir}/train.85-7-8.7.seq2seq.json ${output_dir}/train.85-7-8.8.seq2seq.json \
  --dynet-seed ${seed}

python -m morphology.models.reranker \
  --config-file ${config_file} \
  --train-src data/train.85-7-8.7.src \
  --train-trg data/train.85-7-8.7.trg \
  --train-seq2seq-preds ${output_dir}/train.85-7-8.7.seq2seq.json \
  --valid-src data/dev.src \
  --valid-trg data/dev.trg \
  --valid-seq2seq-preds ${output_dir}/valid.seq2seq.json \
  --test-src data/test.src \
  --test-trg data/test.trg \
  --test-seq2seq-preds ${output_dir}/test.seq2seq.json \
  --unigram-counts data/unigram-counts.txt \
  --vocab-file data/vocab.txt \
  --model-file ${output_dir}/reranker.pt \
  --train-output ${output_dir}/train.85-7-8.7.reranker.json \
  --valid-output ${output_dir}/valid.reranker.json \
  --test-output ${output_dir}/test.reranker.json \
  --metrics-file ${output_dir}/metrics.txt \
  --held-out-srcs data/train.85-7-8.8.src \
  --held-out-trgs data/train.85-7-8.8.trg \
  --held-out-seq2seq-preds ${output_dir}/train.85-7-8.8.seq2seq.json \
  --held-out-outputs ${output_dir}/train.85-7-8.8.reranker.json \
  --dynet-seed ${seed}

python -m morphology.models.ensemble \
  --config-file ${config_file} \
  --train-src data/train.85-7-8.8.src \
  --train-trg data/train.85-7-8.8.trg \
  --train-seq2seq-preds ${output_dir}/train.85-7-8.8.reranker.json \
  --train-dist-hypos ${output_dir}/train.85-7-8.8.distributional.out \
  --valid-src data/dev.src \
  --valid-trg data/dev.trg \
  --valid-seq2seq-preds ${output_dir}/valid.reranker.json \
  --valid-dist-hypos ${output_dir}/valid.distributional.out \
  --test-src data/test.src \
  --test-trg data/test.trg \
  --test-seq2seq-preds ${output_dir}/test.reranker.json \
  --test-dist-hypos ${output_dir}/test.distributional.out \
  --vocab-file data/vocab.txt \
  --model-file ${output_dir}/ensemble.pt \
  --train-output ${output_dir}/train.85-7-8.8.ensemble.out \
  --valid-output ${output_dir}/valid.ensemble.out \
  --test-output ${output_dir}/test.ensemble.out \
  --metrics-file ${output_dir}/metrics.txt \
  --dynet-seed ${seed}

>&2 echo "Finish - `date`"
echo "Finish - `date`"
