#ifndef SUM_SEARCH_H
#define SUM_SEARCH_H

#include "scan_config.cuh"

// Assumes chunk_size is 2*blockDim.x
// Call with NBLOCKS(N, chunk_size)
// Stride should be pow(chunk_size, step), where step is 0-based
// Call repeatedly until stride*chunk_size >= N (ie. next step would only load 1 data point
template<size_t chunk_size, typename T>
__global__ void scan_up_sweep(
    const T* g_idata,       // Input data
    T* g_odata,             // Output data - allocate NBLOCKS(N, chunk_size)*chunk_size
    const size_t N,            // Number of elements in input data
    const size_t end_idx,      // Index read by thread that should load the N-1th element instead of reading off the end
    const size_t stride = 1    // Stride for reading from/writing to idata and odata
) {
  // Can't declare temp directly as s_data since type changes in teplating
  // break compilation. Declare as double2 to get 16 byte memory alignment.
  extern __shared__ double2 temp[];
  T* s_data = reinterpret_cast<T*>(temp);

  int tid = threadIdx.x;
  size_t block_offset = blockIdx.x * chunk_size;
  int offset = 1;

  // Load data into block's shared memory, padding to avoid bank conflicts, and
  // stepping by stride for handling of large arrays
  int ai = tid;
  int bi = tid + chunk_size/2;
  int bank_offset_a = CONFLICT_FREE_OFFSET<sizeof(T)>(ai);
  int bank_offset_b = CONFLICT_FREE_OFFSET<sizeof(T)>(bi);
  bool overwrite_last = false;
  size_t addr_a = (block_offset + ai + 1)*stride - 1;
  size_t addr_b = (block_offset + bi + 1)*stride - 1;
  if (addr_a == end_idx) {
    s_data[ai + bank_offset_a] = g_idata[N-1];
    overwrite_last = true;
  } else if (addr_a < N) s_data[ai + bank_offset_a] = g_idata[addr_a];
  else s_data[ai + bank_offset_a] = 0;
  if (addr_b == end_idx) {
    s_data[bi + bank_offset_b] = g_idata[N-1];
    overwrite_last = true;
  } else if (addr_b < N) s_data[bi + bank_offset_b] = g_idata[addr_b];
  else s_data[bi + bank_offset_b] = 0;


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
  if (addr_a < N) g_odata[addr_a] = s_data[ai + bank_offset_a];
  if (addr_b < N) g_odata[addr_b] = s_data[bi + bank_offset_b];
  // If this block contains the final element (idx N-1), store the total there instead
  if (overwrite_last) {
    int idx = chunk_size - 1;
    idx += CONFLICT_FREE_OFFSET<sizeof(T)>(idx);
    g_odata[N-1] = s_data[idx];
  }
}


// Use 1 block, 1 thread, n_steps * sizeof(T) shared memory
template<size_t chunk_size, typename T>
__global__ void search_tree_device(
  const double r,     // Random number - 0 <= r < 1
  const T* tree,      // Output of up-sweep
  const size_t N,        // Number of elements in the input array
  const size_t begin,    // The final stride from scanning (>= N), divided by 2 until it's < N
  size_t* found          // output -> index of chosen element
) {
  const T rng = r * tree[N-1];

  size_t offset = 0;
  T val_offset = 0;
  for (size_t i = begin; i > 0; i >>= 1) {
    size_t j = offset + i - 1;
    if (j >= N) continue;
    T val = tree[j];
    if (search_cmp(rng, val_offset + val)) {
      val_offset += val;
      offset += i;
    }
  }
  *found = offset;
}

template<typename T>
void sum_search(double r, const T* vec_in_d, T* vec_working_d, const size_t numel, size_t* result_d) {
  static const size_t chunk_size = BLOCK_SIZE*2;
  size_t N = NBLOCKS(numel, chunk_size);
  scan_up_sweep<chunk_size><<<N, BLOCK_SIZE, sizeof(T)*SMEM_PER_BLOCK<sizeof(T)>()>>>(vec_in_d, vec_working_d, numel, numel-1);
  size_t stride = chunk_size;
  for (; stride < numel; stride *= chunk_size) {
    N = NBLOCKS(N, chunk_size);
    scan_up_sweep<chunk_size><<<N, BLOCK_SIZE, sizeof(T)*SMEM_PER_BLOCK<sizeof(T)>()>>>(
          vec_working_d, vec_working_d, numel, ((numel+stride-1)/stride) * stride-1, stride
          );
  }
  size_t begin = stride;
  while (begin >= numel) begin >>= 1;
  search_tree_device<chunk_size><<<1,1>>>(r, vec_working_d, numel, begin, result_d);
}

#endif
