
cmake_minimum_required(VERSION 3.11)
project(DRT2)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)


set(CMAKE_C_COMPILER "gcc-4.2")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-4.2")

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

add_executable(cudart src/main.cu src/camera.h src/hitable.h src/hitable_list.h src/material.h src/cuda_errors.h common/ray.h src/sphere.h common/vec3.h common/options.h)
target_compile_features(cudart PRIVATE cxx_std_11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

# Copy the job scripts
configure_file(job-rt job-rt COPYONLY)
configure_file(job-rt-bench job-rt-bench COPYONLY)
