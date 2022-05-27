#ifndef SCAN_CONFIG_H
#define SCAN_CONFIG_H

#define BLOCK_SIZE 512
#define NBLOCKS(n, THREADS) (((n)+(THREADS)-1)/(THREADS))

#define NUM_BANKS_4B 32
#define LOG_NUM_BANKS_4B 5
#define NUM_BANKS_8B 16
#define LOG_NUM_BANKS_8B 4
/*
#define CONFLICT_FREE_OFFSET(n) \
  ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#define SMEM_PER_BLOCK (BLOCK_SIZE + BLOCK_SIZE/NUM_BANKS + BLOCK_SIZE/(NUM_BANKS*NUM_BANKS))
*/

template<size_t bytes> __host__ __device__
constexpr size_t CONFLICT_FREE_OFFSET(const size_t n);

template<> __host__ __device__
constexpr size_t CONFLICT_FREE_OFFSET<4>(const size_t n) {
   return n >> LOG_NUM_BANKS_4B; //((n) >> NUM_BANKS_4B + (n) >> (2 * LOG_NUM_BANKS_4B));
}
template<> __host__ __device__
constexpr size_t CONFLICT_FREE_OFFSET<8>(const size_t n) {
   return n >> LOG_NUM_BANKS_8B; //((n) >> NUM_BANKS_8B + (n) >> (2 * LOG_NUM_BANKS_8B));
}

template<size_t bytes> __host__ __device__
constexpr size_t SMEM_PER_BLOCK(int data_per_thread=2);

template<> __host__ __device__
constexpr size_t SMEM_PER_BLOCK<4>(int data_per_thread) {
   return data_per_thread*(BLOCK_SIZE + BLOCK_SIZE/NUM_BANKS_4B + BLOCK_SIZE/(NUM_BANKS_4B*NUM_BANKS_4B));
}
template<> __host__ __device__
constexpr size_t SMEM_PER_BLOCK<8>(int data_per_thread) {
   return data_per_thread*(BLOCK_SIZE + BLOCK_SIZE/NUM_BANKS_8B + BLOCK_SIZE/(NUM_BANKS_8B*NUM_BANKS_8B));
}


// Compare function to allow specialisation for floating point numbers
template<typename T> __device__ __host__
inline bool search_cmp(T r, T val) {
  return r > val;
}
// template<> __device__ __host__ 
// inline bool search_cmp<float>(float r, float val) {
//   return r > val && (r - val)/r >= __FLT_EPSILON__;
// }
// template<> __device__ __host__ 
// inline bool search_cmp<double>(double r, double val) {
//   return r > val && (r - val)/r >= __DBL_EPSILON__;
// }


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
