#ifndef CLI_CONFIG_H
#define CLI_CONFIG_H

#include <cstring>
#include <stdexcept>
#include <unordered_set>
#include <cfloat>
#include <climits>

#include "test_result.h"

constexpr static char HELP_MESSAGE[] = 
"Microbenchmarking of sum and search algorithms on GPU.\n" \
"\n" \
"OPTIONS:\n" \
"-h/--help                    Show this message and exit.\n" \
"\n" \
"-s/--seed SEED               Set seed (default 123456789). All benchmarks begin\n" \
"                             from the same seed, so the same data is processed\n" \
"                             by each algorithm.\n" \
"\n" \
"-d/--device ID               Index of GPU to use. Default 0.\n" \
"\n" \
"-n/--num-tests-max MAX       Maximum number of tests to run. Default 100.\n" \
"\n" \
"-nmin/--num-tests-min MIN    Minimum number of tests to run. Default 10.\n" \
"\n" \
"-e/--error-tol TOL           Target standard error of the mean as a fraction\n" \
"                             of 1. Default 0.01 (1%).\n" \
"\n" \
"-N/--num-elements NUM        Number of (randomly generated) elements in the\n" \
"                             test data. Default 10000.\n" \
"\n" \
"-t/--type TYPE               One of i32, i64, f32, or f64 for 32- or 64-bit\n" \
"                             integers or floats, respectively. Default i32.\n" \
"\n" \
"-c/--check                   Check for correctness against the simple linear\n" \
"                             CPU implementation. Any discrepencies are printed\n" \
"                             *before* the benchmark result.\n" \
"\n" \
"-csv                         Display detailed results in plain format that\n" \
"                             can be read as a .csv file.\n"
"\n" \
"-x/--exclude ID [ID [...]]   Exclude tests with the listed 0-based indices.\n" \
"                             Default no tests excluded.\n";

enum class ArrayType {
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
  SCAN_0MEM_CONFLICTS,
  SCAN_0MEM,
  SCAN,
  PARTIAL_0MEM,
  PARTIAL,
  CUB,
  PARTIAL_HALFMEM,
  PARTIAL_HALFMEM_COALESCED,
};

// WARNING: must be in same order as ScanType enum since nvcc
//          doesn't properly support [SCAN]={...} syntax.
constexpr static TableData TABLE_LOOKUP[] = {
  {.proc = "CPU", .scan = "Linear",                         .search = "Linear"},
  {.proc = "CPU", .scan = "Linear, in-place",               .search = "Linear"},
  {.proc = "CPU", .scan = "Linear",                         .search = "Binary"},
  {.proc = "CPU", .scan = "Linear, in-place",               .search = "Binary"},
  {.proc = "CPU", .scan = "Linear, data from GPU",          .search = "Binary"},
  {.proc = "GPU", .scan = "Work efficient with conflicts",  .search = "GPU Binary"},
  {.proc = "GPU", .scan = "Work efficient",                 .search = "GPU Binary"},
  {.proc = "GPU", .scan = "Work efficient, extra memory",   .search = "GPU Binary"},
  {.proc = "GPU", .scan = "Partial",                        .search = "GPU Binary"},
  {.proc = "GPU", .scan = "Partial, extra memory",          .search = "GPU Binary"},
  {.proc = "GPU", .scan = "CUB",                            .search = "GPU Binary"},
  {.proc = "GPU", .scan = "Partial, half output",           .search = "GPU Binary"},
  {.proc = "GPU", .scan = "Partial, half out, coalesced",   .search = "GPU Binary"},
};

// Benchmarking parameters
struct Parameters {
  long seed = 123456789;
  size_t num_tests_max = 100;
  size_t num_tests_min = 10;
  double error_tol = 0.01;
  size_t numel = 10000;
  long double max = INT_MAX;
  int device = 0;
  ArrayType vtype = ArrayType::I32;
  bool check_correctness = false;
  bool csv_format = false;
  std::unordered_set<ScanType> exclude{};
};

template<ScanType scan_type>
TestResult measure_partial_scan(const Parameters &params);

using TestFn = TestResult (*)(const Parameters&);
constexpr static TestFn TESTS[] = {
  &measure_partial_scan<ScanType::CPU_NAIVE>,
  &measure_partial_scan<ScanType::CPU_NAIVE_IN_PLACE>,
  &measure_partial_scan<ScanType::CPU_BINARY>,
  &measure_partial_scan<ScanType::CPU_BINARY_IN_PLACE>,
  &measure_partial_scan<ScanType::CPU_BINARY_WITH_COPY>,
  &measure_partial_scan<ScanType::SCAN_0MEM_CONFLICTS>,
  &measure_partial_scan<ScanType::SCAN_0MEM>,
  &measure_partial_scan<ScanType::SCAN>,
  &measure_partial_scan<ScanType::PARTIAL_0MEM>,
  &measure_partial_scan<ScanType::PARTIAL>,
  &measure_partial_scan<ScanType::CUB>,
  &measure_partial_scan<ScanType::PARTIAL_HALFMEM>,
  &measure_partial_scan<ScanType::PARTIAL_HALFMEM_COALESCED>,
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
    } else if (strcmp(argv[iarg], "-h")==0 || strcmp(argv[iarg], "--help")==0) {
      printf("%s\n", HELP_MESSAGE);
      exit(0);
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
