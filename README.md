README
------

This repository contains microbenchmarking code for various implementations of
GPU algorithms that build and search a prefix sum, where the desired output is
the result of the search (or multiple searches), and not the prefix sum itself.
The focus is on the "partial prefix sum" algorithm ([presented at NCI TechTake
on May 31st, 2022](https://www.youtube.com/watch?v=9QEvmIQnmlw)) in which only
the up-sweep phase of the work-efficient parallel prefix sum is performed, and
the resultant binary tree is searched directly.

Baseline comparisons are the work-efficient parallel prefix sum as described in
[GPU Gems 3](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda),
and the [single pass with decoupled lookback](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back)
algorithm implemented in [CUB](https://github.com/NVIDIA/cub).

A 2x speedup over the work-efficient algorithm is achieved by the partial sum
algorithm, with performance on par with the CUB implementation. Further
optimisation (also applicable to the single pass algorithm), relying on extra
memory working space, provides an additional 20% faster throughput, with the
possibility to go even faster while also requiring less extra global memory (L1
cache size allowing, and at a minor cost in search speed).

LICENSE
-------

Some parts of this code are derived from the source code of the work-efficient
parallel scan algorithm described in GPU Gems 3, and are thereby subject to
copyright as indicated in the relevant files.

All other code in this repository is released under the [MIT license](LICENSE.txt).
