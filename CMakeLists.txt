cmake_minimum_required(VERSION 3.22)
project(cuda_sum_search LANGUAGES CXX CUDA)

# Help info
function(show_build_type_options)
  message("Available build options are:")
  message("  Release          (-O3;-DNDBUG)")
  message("  Debug            (-g3;-ggdb)")
  message("  RelWithDebInfo   (-O3;-g3-ggdb)")
  message("  MinSizeRel       (-Os;-DNDEBUG)")
endfunction()

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
  show_build_type_options()
else()
  message("WARNING: Unknown build type. No optimisation or debug flags will be set.")
  show_build_type_options()
endif()

add_subdirectory(src)

add_custom_target(
  run_tests_i32
  DEPENDS test ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh
  VERBATIM
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh i32
  )

add_custom_target(
  run_tests_i64
  DEPENDS test ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh
  VERBATIM
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh i64
  )

add_custom_target(
  run_tests_f32
  DEPENDS test ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh
  VERBATIM
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh f32
  )

add_custom_target(
  run_tests_f64
  DEPENDS test ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh
  VERBATIM
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.sh f64
  )

