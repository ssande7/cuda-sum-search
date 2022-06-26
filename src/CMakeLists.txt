
set(SRCS
  main.cu
  )

set(INC
  scan.cuh
  sum_search.cuh
  scan_config.cuh
  )

set(CMAKE_CUDA_RUNTIME_LIBRARY Static CACHE STRING "CUDA runtime library")
set(CMAKE_CUDA_STANDARD 17 CACHE STRING "CUDA C++ standard")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(test ${SRCS})

target_compile_options(test PRIVATE
  -acc
  -Wall
  -Wextra
  )

