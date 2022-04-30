#include <cuda.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <stdexcept>
#include <functional>
#include <float.h>

#include "scan_config.cuh"
#include "sum_search.cuh"
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
  long num_tests = 100;
  size_t numel = 10000;
  long double max = INT_MAX;
  int device = 0;
  ArrayType vtype = I32;
  bool check_correctness = false;
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
    } else if (strcmp(argv[iarg], "-n")==0 || strcmp(argv[iarg], "--num-tests")==0) {
      if (++iarg >= argc) throw runtime_error("Missing argument for number of tests");
      params.num_tests = strtol(argv[iarg++], nullptr, 10);
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
    } else throw runtime_error("Unknown argument '" + string(argv[iarg]) + "'");
  }
  if (params.vtype == ArrayType::I32 && params.numel > INT_MAX) throw runtime_error("Number of elements too large to fit in int");
  if (params.vtype == ArrayType::I32 || params.vtype == ArrayType::I64)
    params.max /= params.numel;
  params.max = 100;
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

  void print() {
    static const char* units[] = {"ns", "us", "ms", "s"};
    double div = 1;
    size_t u = 0;
    while (u < 4 && avg_duration / div >= 1000) {
      div *= 1000;
      ++u;
    }
    printf("| %-10s| %-19s| %-14s|% -10g %-2s |\n", 
        header.proc,
        header.scan,
        header.search,
        avg_duration / div, units[u]);
  }
};

enum ScanType {
  SCAN=0,
  SUM_SEARCH,
  CPU_NAIVE,
  CPU_NAIVE_IN_PLACE,
  CPU_BINARY,
  CPU_BINARY_IN_PLACE,
};

// WARNING: must be in same order as enum since nvcc doesn't support [SCAN]={...} syntax.
static const TableData TABLE_LOOKUP[] = {
  {.proc = "GPU", .scan = "Work efficient",   .search = "Binary on GPU"},
  {.proc = "GPU", .scan = "Partial",          .search = "Binary on GPU"},
  {.proc = "CPU", .scan = "Linear",           .search = "Linear"},
  {.proc = "CPU", .scan = "Linear, in-place", .search = "Linear"},
  {.proc = "CPU", .scan = "Linear",           .search = "Binary"},
  {.proc = "CPU", .scan = "Linear, in-place", .search = "Binary"}
};

template<ScanType scan_type, typename DIST>
TestResult test_partial_scan(
    const Parameters &params,
    DIST &dist,
    mt19937 &engine
) {
  typedef typename DIST::result_type T;
  constexpr bool gpu_data = scan_type == SUM_SEARCH || scan_type == SCAN;
  constexpr bool in_place = scan_type == CPU_BINARY_IN_PLACE || scan_type == CPU_NAIVE_IN_PLACE;

  uniform_real_distribution<double> rng_select{}; // gives 0.0 <= r < 1.0

  vector<T> vec(params.numel);
  T* vec_in = nullptr;
  T* vec_out = nullptr;
  size_t result{};
  size_t* result_d = nullptr;
  if (gpu_data) {
    CUDA_CHECK(cudaMalloc((void**)&vec_in,  sizeof(T)*params.numel));
    CUDA_CHECK(cudaMalloc((void**)&vec_out, sizeof(T)*params.numel));
    CUDA_CHECK(cudaMalloc((void**)&result_d, sizeof(size_t)));
  } else {
    vec_out = new T[params.numel];
  }

  clck::duration time_tot{0};
  clck::time_point time_start;
  for (long i = 0; i < params.num_tests; ++i) {
    for (auto &v : vec) v = dist(engine);
    if (gpu_data) CUDA_CHECK(cudaMemcpy(vec_in, vec.data(), sizeof(T)*params.numel, cudaMemcpyHostToDevice));
    if (in_place) memcpy(vec_out, vec.data(), sizeof(T)*params.numel);

    double r = 1.0 - rng_select(engine); // Need 0 < r <= 1

    time_start = clck::now();

    if (scan_type == SCAN) {
      scan_search(r, vec_in, vec_out, params.numel, result_d);
      cudaMemcpy(&result, result_d, sizeof(T), cudaMemcpyDeviceToHost);
    } else if (scan_type == SUM_SEARCH) {
      sum_search(r, vec_in, vec_out, params.numel, result_d);
      cudaMemcpy(&result, result_d, sizeof(T), cudaMemcpyDeviceToHost);
    } else if (scan_type == CPU_NAIVE) {
      result = cpu_naive_scan_search(r, vec, vec_out);
    } else if (scan_type == CPU_NAIVE_IN_PLACE) {
      result = cpu_naive_scan_search_in_place(r, vec_out, params.numel);
    } else if (scan_type == CPU_BINARY) {
      result = cpu_scan_binary_search(r, vec, vec_out);
    } else if (scan_type == CPU_BINARY_IN_PLACE) {
      result = cpu_scan_binary_search_in_place(r, vec_out, params.numel);
    } else {
      fprintf(stderr, "ERROR: unknown algorithm type. This should not occur.");
      exit(-1);
    }

    auto time_end = clck::now();

    if (scan_type != CPU_NAIVE && params.check_correctness) {
      size_t cpu_result = cpu_naive_scan_search_in_place(r, vec.data(), params.numel);
      if (cpu_result != result) printf("Different results! Expected %ld, got %ld\n", cpu_result, result);
    }

    time_tot += time_end - time_start;
  }

  auto test_result = TestResult{
    .header = TABLE_LOOKUP[scan_type],
    .avg_duration = static_cast<double>(time_tot.count()) / params.num_tests
  };

      

  if (gpu_data) {
    CUDA_CHECK(cudaFree(vec_in));
    CUDA_CHECK(cudaFree(vec_out));
    CUDA_CHECK(cudaFree(result_d));
  } else {
    delete [] vec_out;
  }

  return test_result;
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
      return {{.proc="N/A", .scan="UNKNOWN", .search="UNKNOWN"}, 0};
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

  printf("+-----------+--------------------+---------------+--------------+\n"
         "| Processor | Scan Type          | Search Type   | Average Time |\n"
         "+-----------+--------------------+---------------+--------------+\n");
  measure_partial_scan<ScanType::CPU_NAIVE>(params).print();
  measure_partial_scan<ScanType::CPU_NAIVE_IN_PLACE>(params).print();
  measure_partial_scan<ScanType::CPU_BINARY>(params).print();
  measure_partial_scan<ScanType::CPU_BINARY_IN_PLACE>(params).print();
  measure_partial_scan<ScanType::SCAN>(params).print();
  measure_partial_scan<ScanType::SUM_SEARCH>(params).print();
  printf("+-----------+--------------------+---------------+--------------+\n");


  return 0;
}

