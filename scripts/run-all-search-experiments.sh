restarts=30
output_dir=$1
config_file=$2

for i in $(seq 1 ${restarts}); do
  run_dir="${output_dir}/$i"
  mkdir -p ${run_dir}

  name="search_${i}"
  stdout="${run_dir}/stdout"
  stderr="${run_dir}/stderr"

  qsub -N ${name} -o ${stdout} -e ${stderr} scripts/run-search-experiment.sh ${config_file} ${run_dir} ${i}
done
