#pragma ONCE

template<typename T>
__global__ void scan_up_sweep(const T* input, T* output) {

  // TODO: make this work with templating somehow...
  extern __shared__ T temp[];

  int tid = threadIdx.x;
  int block_offset = blockIdx.x * blockDim.x * 2;
  int offset = 1;

  temp[2*tid]   = input[block_offset + 2*tid];
  temp[2*tid+1] = input[block_offset + 2*tid + 1];

  for (int i = blockDim.x; i > 0; i >>= 1) {
    __syncthreads();
    if (tid < i) temp[offset * (2*tid + 2) - 1] += temp[offset * (2*tid + 1) - 1];
    offset *= 2;
  }
  __syncthreads();
  output[block_offset + 2*tid]     = temp[2*tid];
  output[block_offset + 2*tid + 1] = temp[2*tid + 1];
}


// compare function to allow specialisation for floating point numbers
template<typename T> __device__ __host__
inline bool search_cmp(T r, T val) {
  return r > val;
}
template<> __device__ __host__ 
inline bool search_cmp<float>(float r, float val) {
  return r > val && (r - val)/r > 100*__FLT_EPSILON__;
}
template<> __device__ __host__ 
inline bool search_cmp<double>(double r, double val) {
  return r > val && (r - val)/r > 100*__DBL_EPSILON__;
}
// Use 1 block, 1 thread, n_steps * sizeof(T) shared memory
template<typename T>
__global__ void search_tree_device(
  T  r,
  T* tree,
  const int n_steps,
  const int step_scale,
  const int n_padded,
  const int n_tot_padded,
  int* found
) {

  extern __shared__ int step_sz[];
  step_sz[n_steps - 1] = n_padded;
  for (int i = n_steps - 2; i >= 0; --i) {
    step_sz[i] = step_scale * ((step_sz[i+1]/step_scale + step_scale - 1) / step_scale);
  }
  int step_offset = n_tot_padded;
  int sub_offset = 0;

  T val;
  for (int i = 0; i < n_steps; ++i) {
    sub_offset  *= step_scale;
    step_offset -= step_sz[i];

    for (int j = step_scale / 2; j > 0; j >>= 1) {
      val = tree[step_offset + sub_offset + j - 1];
      if (search_cmp(r, val)) {
        r -= val;
        sub_offset += j;
      }
    }
  }
  *found = sub_offset;
}

