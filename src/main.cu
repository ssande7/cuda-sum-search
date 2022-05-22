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

#include "scan_config.cuh"
#include "sum_search.cuh"
#include "sum_search_0mem.cuh"
#include "scan.cuh"
#include "cpu_scan.cuh"

using namespace std;
typedef chrono::high_resolution_clock clck;

// This will be slower than using curand, but we want to check accuracy/speed
// against the naiive linear CPU implementation.
template<typename T>
vector<T> get_random_array(size_t N, T (*rand_T)()) {
  vector<T> vec{};
  vec.reserve(N);
  for (size_t i = 0; i < N; ++i) vec.emplace_back(rand_T());
  return vec;
}

enum ArrayType {
  I32,
  I64,
  F32,
  F64
};

struct Parameters {
  long seed = 123456789;
  size_t num_tests_max = 100;
  size_t num_tests_min = 10;
  double error_tol = 0.01;
  size_t numel = 10000;
  long double max = INT_MAX;
  int device = 0;
  ArrayType vtype = I32;
  bool check_correctness = false;
  bool csv_format = false;
};

// Parse command line arguments
// Throws runtime_error for invalid input
Parameters parse_args(int argc, char** argv) {
  Parameters params{};
  int iarg = 1;
  while (iarg < argc) {
    if (strcmp(argv[iarg], "-s")==0 || strcmp(argv[iarg], "--seed")==0) {
      if (++iarg >= argc) throw runtime_error("Missing argument for seed");
      params.seed = strtol(argv[iarg++], nullptr, 10);
    } else if (strcmp(argv[iarg], "-d")==0 || strcmp(argv[iarg], "--device")==0) {
      if (++iarg >= argc) throw runtime_error("Missing argument for device");
      params.device = static_cast<int>(strtol(argv[iarg++], nullptr, 10));
    } else if (strcmp(argv[iarg], "-n")==0 || strcmp(argv[iarg], "--num-tests-max")==0) {
      if (++iarg >= argc) throw runtime_error("Missing argument for maximum number of tests");
      params.num_tests_max = strtol(argv[iarg++], nullptr, 10);
    } else if (strcmp(argv[iarg], "-nmin")==0 || strcmp(argv[iarg], "--num-tests-min")==0) {
      if (++iarg >= argc) throw runtime_error("Missing argument for minimum number of tests");
      params.num_tests_min = strtol(argv[iarg++], nullptr, 10);
    } else if (strcmp(argv[iarg], "-e")==0 || strcmp(argv[iarg], "--error-tol")==0) {
      if (++iarg >= argc) throw runtime_error("Missing argument for error tolerance");
      params.error_tol = strtod(argv[iarg++], nullptr);
    } else if (strcmp(argv[iarg], "-N")==0 || strcmp(argv[iarg], "--num-elements")==0) {
      if (++iarg >= argc) throw runtime_error("Missing argument for number of elements");
      params.numel = strtol(argv[iarg++], nullptr, 10);
    } else if (strcmp(argv[iarg], "-t")==0 || strcmp(argv[iarg], "--type")==0) {
      if (++iarg >= argc) throw runtime_error("Missing argument for type");
      if (strcmp(argv[iarg], "i32")==0 || strcmp(argv[iarg], "I32")==0) {
        params.vtype = ArrayType::I32;
        params.max = INT_MAX;
      } else if (strcmp(argv[iarg], "i64")==0 || strcmp(argv[iarg], "I64")==0) {
        params.vtype = ArrayType::I64;
        params.max = LONG_MAX;
      } else if (strcmp(argv[iarg], "f32")==0 || strcmp(argv[iarg], "F32")==0) {
        params.vtype = ArrayType::F32;
        params.max = FLT_MAX;
      } else if (strcmp(argv[iarg], "f64")==0 || strcmp(argv[iarg], "F64")==0) {
        params.vtype = ArrayType::F64;
        params.max = DBL_MAX;
      } else throw runtime_error("Unknown data type");
      ++iarg;
    } else if (strcmp(argv[iarg], "-c")==0 || strcmp(argv[iarg], "--check")==0) {
      params.check_correctness = true;
      ++iarg;
    } else if (strcmp(argv[iarg], "-csv")==0) {
      params.csv_format = true;
      ++iarg;
    } else throw runtime_error("Unknown argument '" + string(argv[iarg]) + "'");
  }
  if (params.vtype == ArrayType::I32 && params.numel > INT_MAX) throw runtime_error("Number of elements too large to fit in int");
  if (params.num_tests_min > params.num_tests_max) throw runtime_error("Minimum number of tests must be <= maximum number");
  if (params.vtype == ArrayType::F32) fprintf(stderr, "WARNING: results for large arrays of single precision floats may be inaccurate\n");
  params.max /= params.numel;
  params.max *= 0.8;
  return params;
}

struct TableData {
  const char* proc;
  const char* scan;
  const char* search;
};
struct TestResult {
  TableData header;
  double avg_duration;
  double stdev;
  double err_pc;
  size_t num_tests;

  void print() const {
    static const char* units[] = {"ns", "us", "ms", "s"};
    double div = 1;
    size_t u = 0;
    while (u < 4 && avg_duration / div >= 1000) {
      div *= 1000;
      ++u;
    }
    printf("| %-10s| %-22s| %-14s|% -10g %-2s |\n", 
        header.proc,
        header.scan,
        header.search,
        avg_duration / div, units[u]);
  }
  void print_csv() const {
    printf("%-10s\t%-19s\t%-14s\t%-10g\t%-10g\t%-10g\t%-ld\n",
        header.proc,
        header.scan,
        header.search,
        avg_duration,
        stdev,
        err_pc*100,
        num_tests);
  }
};

enum ScanType {
  CPU_NAIVE=0,
  CPU_NAIVE_IN_PLACE,
  CPU_BINARY,
  CPU_BINARY_IN_PLACE,
  CPU_BINARY_WITH_COPY,
  SCAN,
  PARTIAL_0MEM,
  PARTIAL,
  CUB,
};

// WARNING: must be in same order as enum since nvcc doesn't support [SCAN]={...} syntax.
static const TableData TABLE_LOOKUP[] = {
  {.proc = "CPU", .scan = "Linear",                 .search = "Linear"},
  {.proc = "CPU", .scan = "Linear, in-place",       .search = "Linear"},
  {.proc = "CPU", .scan = "Linear",                 .search = "Binary"},
  {.proc = "CPU", .scan = "Linear, in-place",       .search = "Binary"},
  {.proc = "CPU", .scan = "Linear, data from GPU",  .search = "Binary"},
  {.proc = "GPU", .scan = "Work efficient",         .search = "GPU Binary"},
  {.proc = "GPU", .scan = "Partial",                .search = "GPU Binary"},
  {.proc = "GPU", .scan = "Partial, extra memory",  .search = "GPU Binary"},
  {.proc = "GPU", .scan = "CUB",                    .search = "GPU Binary"}
};

template<ScanType scan_type>
TestResult measure_partial_scan(const Parameters &params);
typedef TestResult (*TestFn)(const Parameters&);

constexpr TestFn TESTS[] = {
  &measure_partial_scan<ScanType::CPU_NAIVE>,
  &measure_partial_scan<ScanType::CPU_NAIVE_IN_PLACE>,
  &measure_partial_scan<ScanType::CPU_BINARY>,
  &measure_partial_scan<ScanType::CPU_BINARY_IN_PLACE>,
  &measure_partial_scan<ScanType::CPU_BINARY_WITH_COPY>,
  &measure_partial_scan<ScanType::SCAN>,
  &measure_partial_scan<ScanType::PARTIAL_0MEM>,
  &measure_partial_scan<ScanType::PARTIAL>,
  &measure_partial_scan<ScanType::CUB>
};

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
                          || scan_type == CUB;
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
      if (cpu_result != result) fprintf(stderr, "Different results! Expected %ld, got %ld from r=%g\n", cpu_result, result, r);
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

  return TestResult{
    .header = TABLE_LOOKUP[scan_type],
    .avg_duration = mean,
    .stdev = sqrt(var),
    .err_pc = err_pc,
    .num_tests = durations.size()
  };
}

template<ScanType scan_type>
TestResult measure_partial_scan(const Parameters &params) {
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

