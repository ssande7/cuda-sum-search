
set(SRCS
  main.cu
  scan.cu
  sum_search.cu
  )

set(INC
  scan.cuh
  sum_search.cuh
  )

set(CMAKE_CUDA_RUNTIME_LIBRARY Static CACHE STRING "CUDA runtime library")
set(CMAKE_CUDA_STANDARD 20 CACHE STRING "CUDA C++ standard")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(test ${SRCS})
