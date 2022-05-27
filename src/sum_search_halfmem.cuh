#ifndef SUM_SEARCH_HALFMEM_H
#define SUM_SEARCH_HALFMEM_H

#include "scan_config.cuh"

namespace partial_half_mem {

// Assumes chunk_size is 2*blockDim.x
// Call with NBLOCKS(N, chunk_size)
// Call repeatedly until chunk_size >= N (ie. next step would only load 1 data point
template<size_t chunk_size, typename T>
__global__ void scan_up_sweep(
    const T* g_idata,  // Input data
    T* g_odata,        // Output data
    T* g_aggregates,   // Storage for aggregate of each block
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
  size_t addr_a = block_offset + 2*ai;
  size_t addr_b = block_offset + 2*bi;
  if (addr_a+1 < N) s_data[ai + bank_offset_a] = g_idata[addr_a] + g_idata[addr_a+1];
  else if (addr_a < N) s_data[ai + bank_offset_a] = g_idata[addr_a];
  else s_data[ai + bank_offset_a] = 0;
  if (addr_b+1 < N) s_data[bi + bank_offset_b] = g_idata[addr_b] + g_idata[addr_b+1];
  else if (addr_b < N) s_data[bi + bank_offset_b] = g_idata[addr_b];
  else s_data[bi + bank_offset_b] = 0;

  //{
  //  int ci = ai + chunk_size;
  //  int di = bi + chunk_size;
  //  int bank_offset_c = CONFLICT_FREE_OFFSET<sizeof(T)>(ci);
  //  int bank_offset_d = CONFLICT_FREE_OFFSET<sizeof(T)>(di);
  //  size_t addr_c = addr_a + chunk_size;
  //  size_t addr_d = addr_b + chunk_size;
  //  if (addr_c < N) s_data[ci + bank_offset_c] = g_idata[addr_c];
  //  else s_data[ci + bank_offset_c] = 0;
  //  if (addr_d < N) s_data[di + bank_offset_d] = g_idata[addr_d];
  //  else s_data[di + bank_offset_d] = 0;
  //  __syncthreads();
  //  int ai1 = 2*ai;
  //  int ai2 = 2*ai+1;
  //  int bi1 = 2*bi;
  //  int bi2 = 2*bi+1;
  //  ai1 += CONFLICT_FREE_OFFSET<sizeof(T)>(ai1);
  //  ai2 += CONFLICT_FREE_OFFSET<sizeof(T)>(ai2);
  //  bi1 += CONFLICT_FREE_OFFSET<sizeof(T)>(bi1);
  //  bi2 += CONFLICT_FREE_OFFSET<sizeof(T)>(bi2);
  //  T a_val1 = s_data[ai1];
  //  T a_val2 = s_data[ai2];
  //  T b_val1 = s_data[bi1];
  //  T b_val2 = s_data[bi2];
  //  __syncthreads();
  //  s_data[ai + bank_offset_a] = a_val1 + a_val2;
  //  s_data[bi + bank_offset_b] = b_val1 + b_val2;
  //}

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
  // printf("%i store: %ld, %ld\n", tid, addr_a, addr_b);

  // Store block aggregates for next stage
  if (tid == blockDim.x - 1) {
    g_aggregates[blockIdx.x] = s_data[bi + bank_offset_b];
    // printf("%i agg: %g @ %p\n", blockIdx.x, static_cast<double>(g_aggregates[blockIdx.x]), g_aggregates + blockIdx.x);
  }
  // Store length of scanned array
  if (tid == 0 && blockIdx.x == 0) {
    *reinterpret_cast<size_t*>(reinterpret_cast<uint8_t*>(g_aggregates)-32) = N;
    // printf("block len: %g\n", static_cast<double>(*reinterpret_cast<size_t*>(reinterpret_cast<uint8_t*>(g_aggregates)-32)));
  }
}


// Use 1 block, 1 thread
template<size_t chunk_size, typename T>
__global__ void search_tree_device(
  const double r,                     // Random number - 0 <= r < 1
  const T*const g_idata, // Original input array
  const T*const tree,    // Output of up-sweep
  const size_t N,                     // Number of elements in the input array
  const T* agg_begin,    // pointer to final level of aggregate tree (chunk_size elements)
  size_t* found          // output -> index of chosen element
) {
  size_t len = *reinterpret_cast<const size_t*>(reinterpret_cast<const uint8_t*>(agg_begin) - 32);
  const T rng = r * *agg_begin;
  size_t offset = 0;
  T val_offset = 0;
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
  scan_up_sweep<chunk_size><<<N, BLOCK_SIZE, sizeof(T)*SMEM_PER_BLOCK<sizeof(T)>()>>>(vec_in_d, vec_working_d, agg_d, numel);
  while (N > 1) {
    size_t this_N = N;
    T* last_agg_d = agg_d;
    T* agg_working_d = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(agg_d) + (N*sizeof(T)+31)/32*32);
    agg_d = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(agg_working_d) + ((N+1)/2*sizeof(T)+31)/32*32 + 32);
    N = NBLOCKS(N, 2*chunk_size);
    scan_up_sweep<chunk_size><<<N, BLOCK_SIZE, sizeof(T)*SMEM_PER_BLOCK<sizeof(T)>()>>>(
          last_agg_d, agg_working_d, agg_d, this_N);
  }
  search_tree_device<chunk_size><<<1,1>>>(r, vec_in_d, vec_working_d, numel, agg_d, result_d);
}

}
#endif
