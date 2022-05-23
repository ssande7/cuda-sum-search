#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o verbose
set -o nounset

if [[ $# -ne 1 ]]; then
    echo "Requires 1 argument for data type"
    exit 1
fi
# Assume run from root of git repository
src_dir="$(pwd)"

# Assume cmake build directory is in scratch
cd ~/scratch/cuda-sum-search/

t="$1"
name=bench_${t}
datetime=$(date +"%F-%H-%M-%S")
logdir=$(pwd)/log

mkdir -p $logdir

squeue --format "%j %t" | grep $name | grep -v ' CG' || sbatch  <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${name}
#SBATCH --nodes=1 --ntasks-per-node=2
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:tesla:1
#SBATCH --output ${logdir}/${name}_${datetime}.stdout
#SBATCH --error ${logdir}/${name}_${datetime}.stderr
set -o errexit
set -o pipefail
set -o verbose

module load \$(cat ${src_dir}/modules_wiener.txt)

cmake --build . --target run_tests_${t}
EOF