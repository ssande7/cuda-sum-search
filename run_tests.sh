#!/usr/bin/env bash
t=$1
max_mem=2 #GB
num_tests_max=5000
len_min=500
len_max=`bc <<< "${max_mem}*1024*1024*1024"`

if [[ "${t}" == i32 ]] || [[ "${t}" == f32 ]]; then
  bytes=4
else
  bytes=8
fi
out_dir=$(pwd)/results/${t}
mkdir -p ${out_dir}
len=${len_min}
while [ `bc <<< "${len}*${bytes}"` -lt ${len_max} ]; do
  fname=`printf %010d ${len}`
  echo Testing $t for N=$len...
  src/test -n ${num_tests_max} -N ${len} -t ${t} -e 0.0025 -csv > ${out_dir}/${fname}.csv
  len=`bc <<< "${len}*2"`
done
