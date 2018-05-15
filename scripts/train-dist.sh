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
  --train-src data/train.src \
  --train-trg data/train.trg \
  --valid-src data/dev.src \
  --valid-trg data/dev.trg \
  --test-src data/test.src \
  --test-trg data/test.trg \
  --word-vectors data/GoogleNews-vectors-negative300.bin.gz \
  --vocab-file data/vocab.txt \
  --model-file ${output_dir}/dist.pt \
  --train-vectors ${output_dir}/train.vecs \
  --train-hypotheses ${output_dir}/train.distributional.out \
  --valid-vectors ${output_dir}/valid.vecs \
  --valid-hypotheses ${output_dir}/valid.distributional.out \
  --test-vectors ${output_dir}/test.vecs \
  --test-hypotheses ${output_dir}/test.distributional.out \
  --metrics-file ${output_dir}/metrics.txt \
  --dynet-seed ${seed}

>&2 echo "Finish - `date`"
echo "Finish - `date`"
