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

enum ScanType {
  SCAN,
  SUM_SEARCH,
};

template<ScanType scan_type, typename DIST>
void test_partial_scan(
    const Parameters &params,
    DIST &dist,
    mt19937 &engine
) {
  typedef typename DIST::result_type T;

  uniform_real_distribution<double> rng_select{};

  vector<T> vec(params.numel); // gives 0.0 <= r < 1.0
  T* vec_in_d = nullptr;  CUDA_CHECK(cudaMalloc((void**)&vec_in_d,  sizeof(T)*params.numel));
  T* vec_out_d = nullptr; CUDA_CHECK(cudaMalloc((void**)&vec_out_d, sizeof(T)*params.numel));
  size_t result{};
  size_t* result_d = nullptr; CUDA_CHECK(cudaMalloc((void**)&result_d, sizeof(size_t)));

  clck::duration time_tot{0};
  clck::time_point time_start;
  for (long i = 0; i < params.num_tests; ++i) {
    for (size_t n = 0; n < params.numel; ++n) vec[n] = 1; //dist(engine);
    CUDA_CHECK(cudaMemcpy(vec_in_d, vec.data(), sizeof(T)*params.numel, cudaMemcpyHostToDevice));
    double r = rng_select(engine);

    time_start = clck::now();

    if (scan_type == SCAN) {
      scan_search(r, vec_in_d, vec_out_d, params.numel, result_d);
    }else if (scan_type == SUM_SEARCH) {
      sum_search(r, vec_in_d, vec_out_d, params.numel, result_d);
    } else {
      fprintf(stderr, "ERROR: unknown algorithm type. This should not occur.");
      exit(-1);
    }
    cudaMemcpy(&result, result_d, sizeof(T), cudaMemcpyDeviceToHost);

    auto time_end = clck::now();

    if (params.check_correctness) {
      for (size_t j = 1; j < params.numel; j++) vec[j] += vec[j-1];
      T rng = r*vec[params.numel-1];
      size_t j = 0;
      while (j < params.numel && rng > vec[j]) ++j;
      if (j != result) printf("Different results! Expected %ld, got %ld\n", j, result);
    }

    time_tot += time_end - time_start;
  }
  auto table_str = "";
  if (scan_type == SCAN) table_str = "Full scan + search";
  else if (scan_type == SUM_SEARCH) table_str = "Partial scan + search";

  printf("%s on GPU\t%10g ns\n", 
      table_str, static_cast<double>(time_tot.count()) / params.num_tests);
}

template<ScanType scan_type>
void measure_partial_scan(const Parameters &params) {
  CUDA_CHECK(cudaSetDevice(params.device));

  mt19937 engine{};
  engine.seed(params.seed);

  switch (params.vtype) {
    case ArrayType::I32:
      {
        uniform_int_distribution<int> dist{0, static_cast<int>(params.max)};
        test_partial_scan<scan_type>(params, dist, engine);
        break;
      }
    case ArrayType::I64:
      {
        uniform_int_distribution<long> dist{0, static_cast<long>(params.max)};
        test_partial_scan<scan_type>(params, dist, engine);
        break;
      }
    case ArrayType::F32:
      {
        uniform_real_distribution<float> dist{0, static_cast<float>(params.max)};
        test_partial_scan<scan_type>(params, dist, engine);
        break;
      }
    case ArrayType::F64:
      {
        uniform_real_distribution<double> dist{0, static_cast<double>(params.max)};
        test_partial_scan<scan_type>(params, dist, engine);
        break;
      }
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

  // measure_partial_scan<ScanType::SCAN>(params);
  measure_partial_scan<ScanType::SUM_SEARCH>(params);


  return 0;
}

