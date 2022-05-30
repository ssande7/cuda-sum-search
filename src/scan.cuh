/*******************************************************************************
NOTE:
The code in the inclusive_scan function is an adaptation of the work-efficient
parallel prefix sum source code described in GPU Gems 3
(https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda),
and is thereby subject to the below copyright statement.

Note that some modifications have been made, which are best seen by comparison
to the original source code.
*******************************************************************************/

/*******************************************************************************
Copyright (c) 2007, NVIDIA CORPORATION. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

For additional information on the license terms, see the CUDA EULA at
https://docs.nvidia.com/cuda/eula/index.html
*******************************************************************************/

#ifndef SCAN_H
#define SCAN_H

#include "scan_config.cuh"

namespace scan_extra_mem {

// Assumes chunk_size = 2*blockDim.x
// Call with NBLOCKS(N, chunk_size)
// Call repeatedly until chunk_size >= N (ie. next step would only load 1 data point)
template<size_t chunk_size, typename T>
__global__ void inclusive_scan(
    const T* g_idata,
    T* g_odata,
    T* g_aggregates,        // Storage for aggregate of each block
    const size_t N
) {
  // Can't declare temp directly as s_data since type changes in teplating
  // break compilation. Declare as double2 to get 16 byte memory alignment.
  extern __shared__ double2 temp[];
  T* s_data = reinterpret_cast<T*>(temp);

  int tid = threadIdx.x;
  size_t block_offset = blockIdx.x * chunk_size;

  int ai = tid;
  int bi = tid + chunk_size/2;
  int bank_offset_a = CONFLICT_FREE_OFFSET<sizeof(T)>(ai);
  int bank_offset_b = CONFLICT_FREE_OFFSET<sizeof(T)>(bi);
  size_t addr_a = block_offset + ai;
  size_t addr_b = block_offset + bi;
  if (addr_a < N) s_data[ai + bank_offset_a] = g_idata[addr_a];
  else s_data[ai + bank_offset_a] = 0;
  if (addr_b < N) s_data[bi + bank_offset_b] = g_idata[addr_b];
  else s_data[bi + bank_offset_b] = 0;
  int offset = 1; 

  // build sum in place up the tree
  for (int d = chunk_size>>1; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
      int ai = offset*(2*tid+1)-1;
      int bi = offset*(2*tid+2)-1;
      ai += CONFLICT_FREE_OFFSET<sizeof(T)>(ai);
      bi += CONFLICT_FREE_OFFSET<sizeof(T)>(bi);
      s_data[bi] += s_data[ai];
    }
    offset *= 2;
  }

  // traverse down tree & build scan
  for (int d = 2; d < chunk_size; d *= 2) {
    offset >>= 1;
    __syncthreads();
    if (tid < d-1) {
      int ai = offset*(tid+1)-1;
      int bi = offset*(tid+1)+(offset>>1)-1;
      ai += CONFLICT_FREE_OFFSET<sizeof(T)>(ai);
      bi += CONFLICT_FREE_OFFSET<sizeof(T)>(bi);
      s_data[bi] += s_data[ai];
    }
  }
  __syncthreads(); 
  if (addr_a < N) g_odata[addr_a] = s_data[ai + bank_offset_a];
  if (addr_b < N) g_odata[addr_b] = s_data[bi + bank_offset_b]; 

  // Store block aggregates for next stage
  if (tid == blockDim.x - 1) g_aggregates[blockIdx.x] = s_data[bi + bank_offset_b];

  // Store length of scanned array
  if (tid == 0 && blockIdx.x == 0)
    *reinterpret_cast<size_t*>(reinterpret_cast<uint8_t*>(g_aggregates)-32) = N;
}

template<size_t chunk_size, typename T>
__global__ void add_tree(T* g_data, const T* g_aggregates, const size_t N) {
  size_t bid = blockIdx.x;
  T val = g_aggregates[bid];
  size_t tid = (bid + 1)*chunk_size + threadIdx.x;
  if (tid < N) g_data[tid] += val;
  tid += blockDim.x;
  if (tid < N) g_data[tid] += val;
}

// Use 1 block, 1 thread, n_steps * sizeof(T) shared memory
template<typename T>
__global__ void binary_search_device(
  const double r,     // Random number - 0 <= r < 1
  const T* tree,      // Output of up-sweep
  const size_t N,     // Number of elements in the input array
  size_t* found       // output -> index of chosen element
) {
  binary_search(r, tree, N, found);
}

template<typename T>
size_t get_extra_mem(size_t N) {
  static const size_t chunk_size = BLOCK_SIZE*2;
  size_t extra = 0;
  while (N > chunk_size) {
    N = NBLOCKS(N, chunk_size);
    extra += (N*sizeof(T) + 31)/32 * 32 + 32;   // Space for previous length. cudaMalloc aligns to 32 byte boundaries
  }
  extra += 64;
  return extra;
}

template<typename T>
void scan_search(double r, const T* vec_in_d, T* vec_working_d, void* extra_mem_d, const size_t numel, size_t* result_d) {
  static const size_t chunk_size = BLOCK_SIZE*2;
  const size_t n_steps = ceil(log2((double)numel)/log2((double)chunk_size));
  size_t sizes[n_steps+1];
  sizes[0] = numel;
  sizes[1] = NBLOCKS(numel, chunk_size);

  T* pointers[n_steps+1];
  pointers[0] = vec_working_d;
  pointers[1] = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(extra_mem_d)+32);
  inclusive_scan<chunk_size><<<sizes[1], BLOCK_SIZE, sizeof(T)*SMEM_PER_BLOCK<sizeof(T)>()>>>(vec_in_d, vec_working_d, pointers[1], numel);

  size_t step = 2;
  while (sizes[step-1] > 1) {
    pointers[step] = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(pointers[step-1]) + 32 + (sizes[step-1]*sizeof(T)+31)/32*32);
    sizes[step] = NBLOCKS(sizes[step-1], chunk_size);
    inclusive_scan<chunk_size><<<sizes[step], BLOCK_SIZE, sizeof(T)*SMEM_PER_BLOCK<sizeof(T)>()>>>(
        pointers[step-1], pointers[step-1], pointers[step], sizes[step-1]);
    ++step;
  }
  step -= 2;
  for (; step > 0; --step) {
    if (sizes[step] > 1) add_tree<chunk_size><<<sizes[step] - 1, BLOCK_SIZE>>>(pointers[step-1], pointers[step], sizes[step-1]);
  }

  binary_search_device<<<1,1>>>(r, vec_working_d, numel, result_d);
}

}

#endif
