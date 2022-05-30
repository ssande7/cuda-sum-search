#ifndef TEST_RESULT_H
#define TEST_RESULT_H

struct TableData {
  const char* proc;
  const char* scan;
  const char* search;
};

struct TestResult {
  static constexpr char table_header[] = 
    "+-----------+-------------------------------+---------------+--------------+\n" \
    "| Processor | Scan Type                     | Search Type   | Average Time |\n" \
    "+-----------+-------------------------------+---------------+--------------+\n";
  static constexpr char table_footer[] =
    "+-----------+-------------------------------+---------------+--------------+\n";
  static constexpr char csv_header[] = 
    "Processor\tScan Type\t\t\tSearch Type\tAvg. Time (ns)\tStd. Dev.\tError (%%)\t# Tests\n";

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
    printf("| %-10s| %-30s| %-14s|% -10g %-2s |\n", 
        header.proc,
        header.scan,
        header.search,
        avg_duration / div, units[u]);
  }
  void print_csv() const {
    printf("%-10s\t%-30s\t%-14s\t%-10g\t%-10g\t%-10g\t%-ld\n",
        header.proc,
        header.scan,
        header.search,
        avg_duration,
        stdev,
        err_pc*100,
        num_tests);
  }
};

#endif
