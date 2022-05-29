This repository contains microbenchmarking code for various implementations of
GPU algorithms that build and search a prefix sum.

Baseline comparisons are the work efficient parallel prefix sum as described in
[GPU Gems 3]{https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda},
and the [single pass dynamic lookback]{https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back}
algorithm implemented in [CUB]{https://github.com/NVIDIA/cub}.
