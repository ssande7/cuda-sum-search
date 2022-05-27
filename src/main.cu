#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <stdexcept>
#include <functional>
#include <float.h>

#include "cli_config.h"
#include "test_result.h"

#include "scan_config.cuh"
#include "sum_search.cuh"
#include "sum_search_0mem.cuh"
#include "sum_search_halfmem.cuh"
#include "scan.cuh"
#include "cpu_scan.cuh"

using namespace std;
typedef chrono::high_resolution_clock clck;

// Run the benchmark
template<ScanType scan_type, typename DIST>
TestResult test_partial_scan(
    const Parameters &params,
    DIST &dist,
    mt19937 &engine
) {
  typedef typename DIST::result_type T;
  constexpr bool gpu_data =  scan_type == PARTIAL_0MEM
                          || scan_type == PARTIAL
                          || scan_type == SCAN
                          || scan_type == CPU_BINARY_WITH_COPY
                          || scan_type == CUB
                          || scan_type == PARTIAL_HALFMEM;
  constexpr bool in_place =  scan_type == CPU_BINARY_IN_PLACE
                          || scan_type == CPU_NAIVE_IN_PLACE;

  uniform_real_distribution<double> rng_select{}; // gives 0.0 <= r < 1.0

  vector<T> vec(params.numel);
  T* vec_in = nullptr;
  T* vec_out = nullptr;
  void* extra_d = nullptr;
  size_t extra_bytes = 0;
  size_t result{};
  size_t* result_d = nullptr;
  if (gpu_data) {
    CUDA_CHECK(cudaMalloc((void**)&vec_in,  sizeof(T)*params.numel));
    CUDA_CHECK(cudaMalloc((void**)&vec_out, sizeof(T)*params.numel));
    CUDA_CHECK(cudaMalloc((void**)&result_d, sizeof(size_t)));
  } else {
    vec_out = new T[params.numel];
  }
  if (scan_type == CPU_BINARY_WITH_COPY) {
    CUDA_CHECK(cudaFree(vec_out));
    vec_out = new T[params.numel];
  }
  // Get storage requirement
  if (scan_type == CUB) {
    cub::DeviceScan::InclusiveSum(extra_d, extra_bytes, vec_in, vec_out, params.numel);
    CUDA_CHECK(cudaMalloc(&extra_d, extra_bytes));
  } else if (scan_type == PARTIAL) {
    extra_bytes = partial_extra_mem::get_extra_mem<T>(params.numel);
    CUDA_CHECK(cudaMalloc(&extra_d, extra_bytes));
  } else if (scan_type == PARTIAL_HALFMEM) {
    extra_bytes = partial_half_mem::get_extra_mem<T>(params.numel);
    CUDA_CHECK(cudaMalloc(&extra_d, extra_bytes));
  }

  clck::duration time_tot{0};
  vector<double> durations{};
  double mean = 0, var = 0, err_pc = 0;
  clck::time_point time_start;
  for (size_t i = 0; i < params.num_tests_max; ++i) {
    for (auto &v : vec) v = dist(engine);
    if (gpu_data) CUDA_CHECK(cudaMemcpy(vec_in, vec.data(), sizeof(T)*params.numel, cudaMemcpyHostToDevice));
    if (in_place) memcpy(vec_out, vec.data(), sizeof(T)*params.numel);

    double r = 1.0 - rng_select(engine); // Need 0 < r <= 1

    time_start = clck::now();

    if (scan_type == SCAN) {
      scan_search(r, vec_in, vec_out, params.numel, result_d);
      cudaMemcpy(&result, result_d, sizeof(T), cudaMemcpyDeviceToHost);
    } else if (scan_type == PARTIAL_0MEM) {
      partial_0mem::sum_search(r, vec_in, vec_out, params.numel, result_d);
      cudaMemcpy(&result, result_d, sizeof(T), cudaMemcpyDeviceToHost);
    } else if (scan_type == PARTIAL) {
      partial_extra_mem::sum_search(r, vec_in, vec_out, extra_d, params.numel, result_d);
      cudaMemcpy(&result, result_d, sizeof(T), cudaMemcpyDeviceToHost);
    } else if (scan_type == PARTIAL_HALFMEM) {
      partial_half_mem::sum_search(r, vec_in, vec_out, extra_d, params.numel, result_d);
      cudaMemcpy(&result, result_d, sizeof(T), cudaMemcpyDeviceToHost);
    } else if (scan_type == CPU_NAIVE) {
      result = cpu_naive_scan_search(r, vec, vec_out);
    } else if (scan_type == CPU_NAIVE_IN_PLACE) {
      result = cpu_naive_scan_search_in_place(r, vec_out, params.numel);
    } else if (scan_type == CPU_BINARY) {
      result = cpu_scan_binary_search(r, vec, vec_out);
    } else if (scan_type == CPU_BINARY_IN_PLACE) {
      result = cpu_scan_binary_search_in_place(r, vec_out, params.numel);
    } else if (scan_type == CPU_BINARY_WITH_COPY) {
      // Include copy time in benchmark
      cudaMemcpy(vec.data(), vec_in, sizeof(T)*params.numel, cudaMemcpyDeviceToHost);
      result = cpu_scan_binary_search(r, vec, vec_out);
    } else if (scan_type == CUB) {
      cub::DeviceScan::InclusiveSum(extra_d, extra_bytes, vec_in, vec_out, params.numel);
      binary_search_device<<<1,1>>>(r, vec_out, params.numel, result_d);
      cudaMemcpy(&result, result_d, sizeof(T), cudaMemcpyDeviceToHost);
    } else {
      fprintf(stderr, "ERROR: unknown algorithm type. This should not occur.");
      exit(-1);
    }

    auto time_end = clck::now();

    if (scan_type != CPU_NAIVE && params.check_correctness) {
      size_t cpu_result = cpu_naive_scan_search_in_place(r, vec.data(), params.numel);
      if (cpu_result != result) {
        fprintf(stderr, "Different results! Expected %ld, got %ld from r=%g\n", cpu_result, result, r);
        exit(1);
      }
    }

    time_tot += time_end - time_start;
    durations.emplace_back(static_cast<double>((time_end - time_start).count()));
    if (i+1 >= params.num_tests_min) {
      mean = static_cast<double>(time_tot.count()) / durations.size();
      var = 0;
      for (auto &v : durations) var += (v - mean)*(v - mean);
      var /= durations.size();
      err_pc = sqrt(var/durations.size()) / mean;
      if (err_pc <= params.error_tol) break;
    }
  }

  if (gpu_data) {
    CUDA_CHECK(cudaFree(vec_in));
    if (scan_type == CPU_BINARY_WITH_COPY)
      delete[] vec_out;
    else
      CUDA_CHECK(cudaFree(vec_out));
    CUDA_CHECK(cudaFree(result_d));
  } else {
    delete [] vec_out;
  }
  if (extra_bytes > 0) CUDA_CHECK(cudaFree(extra_d));

  return TestResult{
    .header = TABLE_LOOKUP[scan_type],
    .avg_duration = mean,
    .stdev = sqrt(var),
    .err_pc = err_pc,
    .num_tests = durations.size()
  };
}

// Wrapper function to handle different random number generators needed for
// different data types
template<ScanType scan_type>
TestResult measure_partial_scan(const Parameters &params) {
  // Exit immediately if excluded
  if (params.exclude.find(scan_type) != params.exclude.end())
    return TestResult{
      .header = TABLE_LOOKUP[scan_type],
      .avg_duration = nan(""),
      .stdev = 0,
      .err_pc = 0,
      .num_tests = 0
    };

  CUDA_CHECK(cudaSetDevice(params.device));

  mt19937 engine{};
  engine.seed(params.seed);

  switch (params.vtype) {
    case ArrayType::I32:
      {
        uniform_int_distribution<int> dist{0, static_cast<int>(params.max)};
        return test_partial_scan<scan_type>(params, dist, engine);
      }
    case ArrayType::I64:
      {
        uniform_int_distribution<long> dist{0, static_cast<long>(params.max)};
        return test_partial_scan<scan_type>(params, dist, engine);
      }
    case ArrayType::F32:
      {
        uniform_real_distribution<float> dist{0, static_cast<float>(params.max)};
        return test_partial_scan<scan_type>(params, dist, engine);
      }
    case ArrayType::F64:
      {
        uniform_real_distribution<double> dist{0, static_cast<double>(params.max)};
        return test_partial_scan<scan_type>(params, dist, engine);
      }
    default:
      return {{.proc="N/A", .scan="UNKNOWN", .search="UNKNOWN"}, 0, 0, 0, 0};
  }
}

int main(int argc, char** argv) {
  // Parse flags
  Parameters params;
  try {
    params = parse_args(argc, argv);
  } catch (runtime_error &e) {
    printf("Error parsing command line flags:\n%s\n", e.what());
    return 1;
  }

  // Run tests
  if (params.csv_format) {
    printf("Processor\tScan Type\t\tSearch Type\tAvg. Time (ns)\tStd. Dev.\tError (%%)\t# Tests\n");
    for (auto t : TESTS) t(params).print_csv();
  } else {
    printf("+-----------+-----------------------+---------------+--------------+\n"
           "| Processor | Scan Type             | Search Type   | Average Time |\n"
           "+-----------+-----------------------+---------------+--------------+\n");
    for (const auto t : TESTS) t(params).print();
    printf("+-----------+-----------------------+---------------+--------------+\n");
  }

  return 0;
}

