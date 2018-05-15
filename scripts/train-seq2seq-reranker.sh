#!/bin/csh
#$ -cwd
#$ -l mem=16G
config_file=$1
output_dir=$2
seed=${3:-4}

echo "Start - `date`"
>&2 echo "Start - `date`"

mkdir -p ${output_dir}

python -m morphology.models.seq2seq \
  --config-file ${config_file} \
  --train-src data/train.90-10.90.src \
  --train-trg data/train.90-10.90.trg \
  --valid-src data/dev.src \
  --valid-trg data/dev.trg \
  --test-src data/test.src \
  --test-trg data/test.trg \
  --vocab-file data/vocab.txt \
  --unigram-counts data/unigram-counts.txt \
  --model-file ${output_dir}/seq2seq.pt \
  --train-output ${output_dir}/train.90-10.90.seq2seq.json \
  --valid-output ${output_dir}/valid.seq2seq.json \
  --test-output ${output_dir}/test.seq2seq.json \
  --held-out-srcs data/train.90-10.10.src \
  --held-out-trgs data/train.90-10.10.trg \
  --held-out-outputs ${output_dir}/train.90-10.10.seq2seq.json \
  --metrics-file ${output_dir}/metrics.txt \
  --dynet-seed ${seed}

python -m morphology.models.reranker \
  --config-file ${config_file} \
  --train-src data/train.90-10.10.src \
  --train-trg data/train.90-10.10.trg \
  --train-seq2seq-preds ${output_dir}/train.90-10.10.seq2seq.json \
  --valid-src data/dev.src \
  --valid-trg data/dev.trg \
  --valid-seq2seq-preds ${output_dir}/valid.seq2seq.json \
  --test-src data/test.src \
  --test-trg data/test.trg \
  --test-seq2seq-preds ${output_dir}/test.seq2seq.json \
  --unigram-counts data/unigram-counts.txt \
  --vocab-file data/vocab.txt \
  --model-file ${output_dir}/reranker.pt \
  --train-output ${output_dir}/train.90-10.10.reranker.json \
  --valid-output ${output_dir}/valid.reranker.json \
  --test-output ${output_dir}/test.reranker.json \
  --metrics-file ${output_dir}/metrics.txt \
  --dynet-seed ${seed}

>&2 echo "Finish - `date`"
echo "Finish - `date`"
