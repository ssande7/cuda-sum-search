/*******************************************************************************
NOTE:
The code in the scan_up_sweep function is derived from sections of the
work-efficient parallel prefix sum source code described in GPU Gems 3
(https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda),
and may thereby be subject to the below copyright statement.
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

#ifndef SUM_SEARCH_HALFMEM_COALESCED_H
#define SUM_SEARCH_HALFMEM_COALESCED_H

#include "scan_config.cuh"

namespace partial_half_mem_coalesced {

// Assumes chunk_size is 2*blockDim.x
// Call with NBLOCKS(N, 2*chunk_size)
// Call repeatedly until chunk_size >= N (ie. next step would only load 1 data point)
template<size_t chunk_size, typename T>
__global__ void scan_up_sweep(
    const T* __restrict__ g_idata,  // Input data
    T* __restrict__ g_odata,        // Output data
    T* __restrict__ g_aggregates,   // Storage for aggregate of each block
    const size_t N                  // Number of elements in input data
) {
  // Can't declare temp directly as s_data since type changes in teplating
  // break compilation. Declare as double2 to get 16 byte memory alignment.
  extern __shared__ double2 temp[];
  T* s_data = reinterpret_cast<T*>(temp);

  int tid = threadIdx.x;
  size_t block_offset = blockIdx.x * 2 * chunk_size;
  int offset = 1;

  // Load data into block's shared memory, padding to avoid bank conflicts
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

  {
    int ci = ai + chunk_size;
    int di = bi + chunk_size;
    int bank_offset_c = CONFLICT_FREE_OFFSET<sizeof(T)>(ci);
    int bank_offset_d = CONFLICT_FREE_OFFSET<sizeof(T)>(di);
    size_t addr_c = addr_a + chunk_size;
    size_t addr_d = addr_b + chunk_size;
    if (addr_c < N) s_data[ci + bank_offset_c] = g_idata[addr_c];
    else s_data[ci + bank_offset_c] = 0;
    if (addr_d < N) s_data[di + bank_offset_d] = g_idata[addr_d];
    else s_data[di + bank_offset_d] = 0;

    // Store 2 sets of consecutive values in registers
    __syncthreads();
    int ai1 = 2*ai;
    int ai2 = 2*ai+1;
    int bi1 = 2*bi;
    int bi2 = 2*bi+1;
    ai1 += CONFLICT_FREE_OFFSET<sizeof(T)>(ai1);
    ai2 += CONFLICT_FREE_OFFSET<sizeof(T)>(ai2);
    bi1 += CONFLICT_FREE_OFFSET<sizeof(T)>(bi1);
    bi2 += CONFLICT_FREE_OFFSET<sizeof(T)>(bi2);
    T a_val1 = s_data[ai1];
    T a_val2 = s_data[ai2];
    T b_val1 = s_data[bi1];
    T b_val2 = s_data[bi2];

    // Store sums in shared memory ready for scan
    __syncthreads();
    s_data[ai + bank_offset_a] = a_val1 + a_val2;
    s_data[bi + bank_offset_b] = b_val1 + b_val2;
  }

#pragma unroll
  for (int d = chunk_size/2; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
      int ai_l = offset*(2*tid+1)-1;
      int bi_l = offset*(2*tid+2)-1;
      ai_l += CONFLICT_FREE_OFFSET<sizeof(T)>(ai_l);
      bi_l += CONFLICT_FREE_OFFSET<sizeof(T)>(bi_l);
      s_data[bi_l] += s_data[ai_l];
    }
    offset *= 2;
  }
  __syncthreads();
  // Assume sufficient space is allocated
  addr_a = blockIdx.x*chunk_size + ai;
  addr_b = blockIdx.x*chunk_size + bi;
  if (addr_a < (N+1)/2) g_odata[addr_a] = s_data[ai + bank_offset_a];
  if (addr_b < (N+1)/2) g_odata[addr_b] = s_data[bi + bank_offset_b];

  // Store block aggregates for next stage
  if (tid == blockDim.x - 1) g_aggregates[blockIdx.x] = s_data[bi + bank_offset_b];

  // Store length of scanned array
  if (tid == 0 && blockIdx.x == 0)
    *reinterpret_cast<size_t*>(reinterpret_cast<uint8_t*>(g_aggregates)-32) = N;
}


// Use 1 block, 1 thread
template<size_t chunk_size, typename T>
__global__ void search_tree_device(
  const double r,                     // Random number - 0 <= r < 1
  const T*const __restrict__ g_idata, // Original input array
  const T*const __restrict__ tree,    // Output of up-sweep
  const size_t N,                     // Number of elements in the input array
  const T* __restrict__ agg_begin,    // pointer to final level of aggregate tree (chunk_size elements)
  size_t* __restrict__ found          // output -> index of chosen element
) {
  size_t len = *reinterpret_cast<const size_t*>(reinterpret_cast<const uint8_t*>(agg_begin) - 32);
  const T rng = r * *agg_begin;
  size_t offset = 0;
  T val_offset = 0;

  // NOTE: this could potentially be sped up by pre-fetching once the range of
  //       possible values is within 1 cache line
  while (len < N) {
    agg_begin = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(agg_begin) - 32 - ((len+1)/2*sizeof(T)+31)/32*32);
#pragma unroll
    for (size_t i = chunk_size >> 1; i > 0; i >>= 1) {
      size_t j = offset + i - 1;
      if (j >= (len+1)/2) continue;
      T val = agg_begin[j];
      if (search_cmp(rng, val_offset + val)) {
        val_offset += val;
        offset += i;
      }
    }
    // Search last part of list
    agg_begin = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(agg_begin) - (len*sizeof(T)+31)/32*32);
    offset *= 2;
    if (offset < len) {
      T val = agg_begin[offset];
      if (search_cmp(rng, val_offset + val)) {
        val_offset += val;
        offset += 1;
      }
    }
    // Move to next
    len = *reinterpret_cast<const size_t*>((const uint8_t*)agg_begin - 32);
    offset *= chunk_size;
  }

#pragma unroll
  for (size_t i = chunk_size >> 1; i > 0; i >>= 1) {
    size_t j = offset + i - 1;
    if (j >= (N+1)/2) continue;
    T val = tree[j];
    if (search_cmp(rng, val_offset + val)) {
      val_offset += val;
      offset += i;
    }
  }
  offset *= 2;
  if (offset < N && search_cmp(rng, val_offset + g_idata[offset])) offset += 1;
  *found = offset;
}

template<typename T>
size_t get_extra_mem(size_t N) {
  static const size_t chunk_size = BLOCK_SIZE*2;
  size_t extra = 0;
  while (N > 2*chunk_size) {
    N = NBLOCKS(N, 2*chunk_size);
    extra += ((N+1)/2*sizeof(T) + 31)/32 * 32 + (N*sizeof(T) + 31)/32 * 32 + 32;   // Space for previous length. cudaMalloc aligns to 32 byte boundaries
  }
  extra += 64; // Space for final total
  return extra;
}

template<typename T>
void sum_search(double r, const T* vec_in_d, T* vec_working_d, void* extra_mem_d, const size_t numel, size_t* result_d) {
  static const size_t chunk_size = BLOCK_SIZE*2;
  size_t N = NBLOCKS(numel, 2*chunk_size);
  T* agg_d = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(extra_mem_d)+32);
  scan_up_sweep<chunk_size><<<N, BLOCK_SIZE, sizeof(T)*SMEM_PER_BLOCK<sizeof(T)>(4)>>>(vec_in_d, vec_working_d, agg_d, numel);
  while (N > 1) {
    size_t this_N = N;
    T* last_agg_d = agg_d;
    T* agg_working_d = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(agg_d) + (N*sizeof(T)+31)/32*32);
    agg_d = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(agg_working_d) + ((N+1)/2*sizeof(T)+31)/32*32 + 32);
    N = NBLOCKS(N, 2*chunk_size);
    scan_up_sweep<chunk_size><<<N, BLOCK_SIZE, sizeof(T)*SMEM_PER_BLOCK<sizeof(T)>(4)>>>(
          last_agg_d, agg_working_d, agg_d, this_N);
  }
  search_tree_device<chunk_size><<<1,1>>>(r, vec_in_d, vec_working_d, numel, agg_d, result_d);
}

}
#endif
