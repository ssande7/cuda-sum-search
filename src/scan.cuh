#ifndef SCAN_H
#define SCAN_H

#include "scan_config.cuh"

// Assumes chunk_size = 2*blockDim.x
template<size_t chunk_size, typename T>
__global__ void inclusive_scan(
    const T* g_idata,
    T* g_odata,
    const size_t N,
    const size_t end_idx,
    const size_t stride = 1
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

  /* if (tid==0) */
  /* s_data[chunk_size - 1 + CONFLICT_FREE_OFFSET(chunk_size - 1)] = 0; */
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

      /* float t = s_data[ai]; */
      /* s_data[ai] = s_data[bi]; */
      /* s_data[bi] += t; */
    }
  }
  __syncthreads(); 
  if (addr_a < N) g_odata[addr_a] = s_data[ai + bank_offset_a];
  if (addr_b < N) g_odata[addr_b] = s_data[bi + bank_offset_b]; 
  if (overwrite_last) {
    int idx = chunk_size - 1;
    idx += CONFLICT_FREE_OFFSET<sizeof(T)>(idx);
    g_odata[N-1] = s_data[idx];
  }
}

template<size_t chunk_size, typename T>
__global__ void add_tree(T* g_data, const size_t N, const size_t stride) {
  size_t bid = stride*(blockIdx.x + 1)-1;
  T val = g_data[bid];
  size_t tid = bid + (threadIdx.x + 1)*(stride/chunk_size);
  if (tid < N-1) g_data[tid] += val;
  tid += stride/2;
  if (tid < N-1 && threadIdx.x < blockDim.x-1) g_data[tid] += val;
}

template<typename T>
__host__ __device__ inline void binary_search(
    const double r,
    const T* tree,
    const size_t N,
    size_t *found
) {
  const T rng = r * tree[N-1];
  size_t offset = 0, max = N-1, i;
  while (offset <= max) {
    i = offset + (max - offset) / 2;
    if (search_cmp(rng, tree[i])) {
      offset = i + 1;
    } else {
      max = i - 1;
    }
  }
  *found = offset;
}

// Use 1 block, 1 thread, n_steps * sizeof(T) shared memory
template<typename T>
__global__ void binary_search_device(
  const double r,     // Random number - 0 <= r < 1
  const T* tree,      // Output of up-sweep
  const size_t N,        // Number of elements in the input array
  size_t* found          // output -> index of chosen element
) {
  binary_search(r, tree, N, found);
}

template<typename T>
void scan_search(double r, const T* vec_in_d, T* vec_working_d, const size_t numel, size_t* result_d) {
  static const size_t chunk_size = BLOCK_SIZE*2;
  const size_t n_steps = ceil(log2((double)numel)/log2((double)chunk_size));
  size_t sizes[n_steps+1];
  sizes[0] = numel;
  sizes[1] = NBLOCKS(numel, chunk_size);

  inclusive_scan<chunk_size><<<sizes[1], BLOCK_SIZE, sizeof(T)*SMEM_PER_BLOCK<sizeof(T)>()>>>(vec_in_d, vec_working_d, numel, numel-1);

  size_t stride = chunk_size;
  size_t step = 2;
  for (; stride < numel; stride *= chunk_size) {
    sizes[step] = NBLOCKS(sizes[step-1], chunk_size);
    inclusive_scan<chunk_size><<<sizes[step], BLOCK_SIZE, sizeof(T)*SMEM_PER_BLOCK<sizeof(T)>()>>>(
        vec_working_d, vec_working_d, numel, ((numel+stride-1)/stride) * stride-1, stride);
    ++step;
  }
  step -= 2;
  stride /= chunk_size;
  for (; step > 0; --step, stride /= chunk_size) {
    if (sizes[step] > 1) add_tree<chunk_size><<<sizes[step] - 1, BLOCK_SIZE>>>(vec_working_d, numel, stride);
  }

  binary_search_device<<<1,1>>>(r, vec_working_d, numel, result_d);
}

#endif
