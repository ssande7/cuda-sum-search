#ifndef SCAN_CONFIG_H
#define SCAN_CONFIG_H

#define BLOCK_SIZE 512
#define NBLOCKS(n, THREADS) (((n)+(THREADS)-1)/(THREADS))

#define NUM_BANKS_4B 32
#define LOG_NUM_BANKS_4B 5
#define NUM_BANKS_8B 16
#define LOG_NUM_BANKS_8B 4

template<size_t bytes> __host__ __device__ __forceinline__
constexpr size_t CONFLICT_FREE_OFFSET(const size_t n);

template<> __host__ __device__ __forceinline__
constexpr size_t CONFLICT_FREE_OFFSET<4>(const size_t n) {
   return n >> LOG_NUM_BANKS_4B;
}
template<> __host__ __device__ __forceinline__
constexpr size_t CONFLICT_FREE_OFFSET<8>(const size_t n) {
   return n >> LOG_NUM_BANKS_8B;
}

template<size_t bytes> __host__ __device__ __forceinline__
constexpr size_t SMEM_PER_BLOCK(int data_per_thread=2);

template<> __host__ __device__ __forceinline__
constexpr size_t SMEM_PER_BLOCK<4>(int data_per_thread) {
   return data_per_thread*(BLOCK_SIZE + BLOCK_SIZE/NUM_BANKS_4B +
       BLOCK_SIZE/(NUM_BANKS_4B*NUM_BANKS_4B));
}
template<> __host__ __device__ __forceinline__
constexpr size_t SMEM_PER_BLOCK<8>(int data_per_thread) {
   return data_per_thread*(BLOCK_SIZE + BLOCK_SIZE/NUM_BANKS_8B +
       BLOCK_SIZE/(NUM_BANKS_8B*NUM_BANKS_8B));
}


// Compare function to allow specialisation for floating point
// numbers/other types if necessary
template<typename T> __device__ __host__ __forceinline__
bool search_cmp(T r, T val) {
  return r > val;
}

template<typename T>
__host__ __device__ __forceinline__
void binary_search(
    const double r,
    const T* tree,
    const size_t N,
    size_t *found
) {
  const T rng = r * tree[N-1];
  size_t offset = 0, max = N-1, i;
  while (offset < max) {
    i = offset + (max - offset) / 2;
    if (search_cmp(rng, tree[i])) {
      offset = i + 1;
    } else if (i > 0 && search_cmp(rng, tree[i-1])) {
      offset = i;
      break;
    } else {
      max = i;
    }
  }
  *found = offset;
}



#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t status = call;                                                \
    if (status != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error ('%s':%i): %s.\n", __FILE__,                \
              __LINE__, cudaGetErrorString(status));                          \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  } while (0)

#endif
