#ifndef CLI_CONFIG_H
#define CLI_CONFIG_H

#include <stdexcept>
#include <unordered_set>

#include "test_result.h"

enum ArrayType {
  I32,
  I64,
  F32,
  F64
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
  NONE        // Placeholder for iteration
};

// WARNING: must be in same order as enum since nvcc doesn't support [SCAN]={...} syntax.
constexpr TableData TABLE_LOOKUP[] = {
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
  std::unordered_set<ScanType> exclude{};
};

template<ScanType scan_type>
TestResult measure_partial_scan(const Parameters &params);

using TestFn = TestResult (*)(const Parameters&);
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

// Parse command line arguments
// Throws runtime_error for invalid input
Parameters parse_args(int argc, char** argv) {
  using namespace std;
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
    } else if (strcmp(argv[iarg], "-x")==0 || strcmp(argv[iarg], "--exclude")==0) {
      if (++iarg >= argc) throw runtime_error("Missing arguments for -x/--exclude");
      while (iarg < argc) {
        if (argv[iarg][0] == '-') break;
        params.exclude.insert(static_cast<ScanType>(strtol(argv[iarg++], nullptr, 10)));
      }
    } else throw runtime_error("Unknown argument '" + string(argv[iarg]) + "'");
  }
  if (params.vtype == ArrayType::I32 && params.numel > INT_MAX) throw runtime_error("Number of elements too large to fit in int");
  if (params.num_tests_min > params.num_tests_max) throw runtime_error("Minimum number of tests must be <= maximum number");
  if (params.vtype == ArrayType::F32) fprintf(stderr, "WARNING: results for large arrays of single precision floats may be inaccurate\n");
  params.max /= params.numel;
  params.max *= 0.8;
  return params;
}

#endif
