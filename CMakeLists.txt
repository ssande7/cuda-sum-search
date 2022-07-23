cmake_minimum_required(VERSION 3.18)
project(cuda_sum_search LANGUAGES CXX CUDA)


if (CMAKE_BUILD_TYPE MATCHES Debug)
  message("Building in Debug mode. Be sure to reconfigure with CMAKE_BUILD_TYPE=Release for production.")
elseif (CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message("Building in Release mode with debug symbols.")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
  message("Building in Release mode.")
elseif (CMAKE_BUILD_TYPE MATCHES MinSizeRel)
  message("Building in Release mode with minimum size optimisation.")
elseif (CMAKE_BUILD_TYPE MATCHES "")
  message("WARNING: No build type specified. Defaulting to Release.")
  set(CMAKE_BUILD_TYPE Release)
else()
  message("WARNING: Unknown build type!")
endif()

set(BLOCK_SIZES 32;64;128;256;512)

add_subdirectory(src)

foreach(BLOCK_SIZE IN LISTS BLOCK_SIZES)
  add_custom_target(
    run_tests_${BLOCK_SIZE}_i32
    DEPENDS test_${BLOCK_SIZE} ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh
    VERBATIM
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh ${BLOCK_SIZE} i32
    )

  add_custom_target(
    run_tests_${BLOCK_SIZE}_i64
    DEPENDS test_${BLOCK_SIZE} ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh
    VERBATIM
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh ${BLOCK_SIZE} i64
    )

  add_custom_target(
    run_tests_${BLOCK_SIZE}_f32
    DEPENDS test_${BLOCK_SIZE} ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh
    VERBATIM
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh ${BLOCK_SIZE} f32
    )

  add_custom_target(
    run_tests_${BLOCK_SIZE}_f64
    DEPENDS test_${BLOCK_SIZE} ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh
    VERBATIM
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh ${BLOCK_SIZE} f64
    )
endforeach()
