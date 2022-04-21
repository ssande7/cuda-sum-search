#ifndef SCAN_H
#define SCAN_H

#include "scan_config.cuh"

// Assumes chunk_size = 2*blockDim.x
template<size_t chunk_size, typename T>
__global__ void inclusive_scan(const T* g_idata, T* g_odata, const int N, const int stride = 1) {
  // Can't declare temp directly as s_data since type changes in teplating
  // break compilation. Declare as double2 to get 16 byte memory alignment.
  extern __shared__ double2 temp[];
  T* s_data = reinterpret_cast<T*>(temp);

  int thid = threadIdx.x;
  size_t block_offset = blockIdx.x * chunk_size;

  int ai = thid;
  int bi = thid + chunk_size/2;
  int bank_offset_a = CONFLICT_FREE_OFFSET<sizeof(T)>(ai);
  int bank_offset_b = CONFLICT_FREE_OFFSET<sizeof(T)>(bi);
  if (block_offset + ai < N) s_data[ai + bank_offset_a] = g_idata[(block_offset + ai)*stride];
  else s_data[ai + bank_offset_a] = 0;
  if (block_offset + bi < N) s_data[bi + bank_offset_b] = g_idata[(block_offset + bi)*stride];
  else s_data[bi + bank_offset_b] = 0;
  int offset = 1; 

  // build sum in place up the tree
  for (int d = chunk_size>>1; d > 0; d >>= 1) {
    __syncthreads();
    if (thid < d) {
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      ai += CONFLICT_FREE_OFFSET<sizeof(T)>(ai);
      bi += CONFLICT_FREE_OFFSET<sizeof(T)>(bi);
      s_data[bi] += s_data[ai];
    }
    offset *= 2;
  }

  /* if (thid==0) */
  /* s_data[chunk_size - 1 + CONFLICT_FREE_OFFSET(chunk_size - 1)] = 0; */
  // traverse down tree & build scan
  for (int d = 2; d < chunk_size; d *= 2) {
    offset >>= 1;
    __syncthreads();
    if (thid < d-1) {
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+1)+(offset>>1)-1;
      ai += CONFLICT_FREE_OFFSET<sizeof(T)>(ai);
      bi += CONFLICT_FREE_OFFSET<sizeof(T)>(bi);
      s_data[bi] += s_data[ai];

      /* float t = s_data[ai]; */
      /* s_data[ai] = s_data[bi]; */
      /* s_data[bi] += t; */
    }
  }  
  __syncthreads(); 
  if ((block_offset + ai)*stride < N) g_odata[(block_offset + ai)*stride] = s_data[ai + bank_offset_a];
  if ((block_offset + bi)*stride < N) g_odata[(block_offset + bi)*stride] = s_data[bi + bank_offset_b]; 
}

// Use 1 block, 1 thread, n_steps * sizeof(T) shared memory
template<size_t chunk_size, typename T>
__global__ void binary_search_device(
  const double r,     // Random number - 0 <= r < 1
  const T* tree,      // Output of up-sweep
  const size_t N,        // Number of elements in the input array
  size_t* found          // output -> index of chosen element
) {
  T rng = r * tree[N-1];

  size_t offset = 0;
  for (int i = (N+1)>>1; i > 0; i >>= 1) {
    T val = tree[offset + i - 1];
    if (rng > val) offset += i;
  }
  *found = offset;
}

template<typename T>
void scan_search(double r, const T* vec_in_d, T* vec_working_d, const size_t numel, size_t* result_d) {
  static const size_t chunk_size = BLOCK_SIZE*2;
  size_t N = NBLOCKS(numel, chunk_size);
  inclusive_scan<chunk_size><<<N, BLOCK_SIZE, sizeof(T)*SMEM_PER_BLOCK<sizeof(T)>()>>>(vec_in_d, vec_working_d, N);
  size_t stride = chunk_size;
  for (; stride < numel; stride *= chunk_size) {
    N = NBLOCKS(N, chunk_size);
    inclusive_scan<chunk_size><<<N, BLOCK_SIZE, sizeof(T)*SMEM_PER_BLOCK<sizeof(T)>()>>>(vec_working_d, vec_working_d, N, stride);
  }
  size_t begin = stride;
  while (begin >= N) begin >>= 1;
  binary_search_device<chunk_size><<<1,1>>>(r, vec_working_d, numel, result_d);
}

#endif
