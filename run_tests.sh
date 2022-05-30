#!/usr/bin/env bash
# Run from build directory

t=$1 # i32, i64, f32, or f64
max_mem=6 #GB
num_tests_max=5000
len_min=500
len_max=$(( ${max_mem}*1024*1024*1024 ))
max_slow=$(( 128*1024*1024 ))
max_cpu=$(( 256*1024*1024 ))

if [[ "${t}" == i32 ]] || [[ "${t}" == f32 ]]; then
  bytes=4
else
  bytes=8
fi
out_dir=$(pwd)/results/${t}
mkdir -p ${out_dir}
len=${len_min}
while [ $(( ${len}*${bytes} )) -lt ${len_max} ]; do
  fname=`printf %010d ${len}`
  echo Testing $t for N=$len...
  if [ ! -f ${out_dir}/${fname}.csv ]; then
    if [ $(( ${len}*${bytes} )) -lt ${max_cpu} ]; then
      exclude=""
    elif [ $(( ${len}*${bytes} )) -lt ${max_slow} ]; then
      exclude=(-x 4)
    else
      exclude=(-x 0 1 2 3 4)
    fi
    src/test -n ${num_tests_max} -N ${len} -t ${t} ${exclude[@]} -e 0.0025 -csv > ${out_dir}/${fname}.csv
  fi
  len=$(( ${len}*2 ))
  num_tests_max=$(( ${num_tests_max}*8/10 ))
done
