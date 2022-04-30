#ifndef CPU_SCAN_H
#define CPU_SCAN_H

#include <vector>
#include "scan.cuh"

template<typename T>
size_t cpu_naive_scan_search(const double r, const std::vector<T> &vec_in, T* vec_out) {
  vec_out[0] = vec_in[0];
  for (size_t j = 1; j < vec_in.size(); ++j) vec_out[j] = vec_out[j-1] + vec_in[j];
  T rng = r*vec_out[vec_in.size()-1];
  size_t j = 0;
  while (j < vec_in.size() && rng > vec_out[j]) ++j;
  return j;
}
template<typename T>
size_t cpu_naive_scan_search_in_place(const double r, T* vec, const size_t N) {
  for (size_t j = 1; j < N; ++j) vec[j] += vec[j-1];
  T rng = r*vec[N-1];
  size_t j = 0;
  while (j < N && rng > vec[j]) ++j;
  return j;
}

template<typename T>
size_t cpu_scan_binary_search(const double r, const std::vector<T> &vec_in, T* vec_out) {
  vec_out[0] = vec_in[0];
  for (size_t j = 1; j < vec_in.size(); ++j) vec_out[j] = vec_out[j-1] + vec_in[j];
  size_t out;
  binary_search(r, vec_out, vec_in.size(), &out);
  return out;
}

template<typename T>
size_t cpu_scan_binary_search_in_place(const double r, T* vec, const size_t N) {
  for (size_t j = 1; j < N; ++j) vec[j] += vec[j-1];
  size_t out;
  binary_search(r, vec, N, &out);
  return out;
}


#endif
