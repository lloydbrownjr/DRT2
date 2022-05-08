
cmake_minimum_required(VERSION 3.11)
project(DRT2)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

find_package(MPI REQUIRED)

add_executable(cudart src/main.cu)
target_link_libraries(cudart PRIVATE MPI::MPI_CXX)
target_compile_features(cudart PRIVATE cxx_std_11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

# Copy the job scripts
configure_file(job-rt job-rt COPYONLY)
configure_file(job-rt-bench job-rt-bench COPYONLY)
