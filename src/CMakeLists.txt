
set(SRCS
  main.cu
  )

set(INC
  scan.cuh
  sum_search.cuh
  scan_config.cuh
  )

set(CMAKE_CUDA_RUNTIME_LIBRARY Static CACHE STRING "CUDA runtime library")
set(CMAKE_CUDA_STANDARD 20 CACHE STRING "CUDA C++ standard")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

foreach(BLOCK_SIZE IN LISTS BLOCK_SIZES)
  add_executable(test_${BLOCK_SIZE} ${SRCS})
  target_compile_definitions(test_${BLOCK_SIZE} PUBLIC BLOCK_SIZE=${BLOCK_SIZE})
  target_compile_options(test_${BLOCK_SIZE} PRIVATE
    -Wall
    -Wextra
    )
endforeach()
