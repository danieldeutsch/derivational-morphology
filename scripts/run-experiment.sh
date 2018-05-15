experiment_dir=$1
config_file=$2

num_runs=30
for i in $(seq 1 ${num_runs}); do
  for model in dist seq2seq seq2seq-reranker dist-seq2seq dist-seq2seq-reranker; do
    output_dir="${experiment_dir}/${model}/${i}"
    command="scripts/train-${model}.sh ${config_file} ${output_dir} ${i}"

    if [[ $* == *--local* ]]; then
      sh ${command}
    else
      name="train-${model}-${i}"
      stdout=${output_dir}/stdout
      stderr=${output_dir}/stderr
      # The NLP grid requires the path to stdout and stderr to
      # exist before the job starts
      mkdir -p ${output_dir}
	
      qsub -N ${name} -o ${stdout} -e ${stderr} ${command}
    fi
  done
done
