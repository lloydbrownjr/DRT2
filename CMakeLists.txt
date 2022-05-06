
cmake_minimum_required(VERSION 3.11)
project(DRT2)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)


set(CMAKE_C_COMPILER "mpicc")
set(CMAKE_CXX_COMPILER "mpicxx")

find_package(LibXml2 REQUIRED)

include_directories(SYSTEM /opt/packages/mvapich2/intel/2.3.5-intel20.4/include)
set(MPI_PREFIX "/opt/packages/mvapich2-2.3.5/gcc-8.3.1")
# set(MPI_PREFIX "/opt/packages/openmpi/gcc/4.0.2-gcc8.3.1")
set(MPI_INCLUDE_DIR "${MPI_PREFIX}/include")
set(MPI_SHARED_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}mpi")
set(MPI_LIBRARIES "${MPI_PREFIX}/lib/${MPI_SHARED_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}")
include_directories(SYSTEM ${MPI_INCLUDE_DIR})

# We require the GNU compiler for thisignment.s as
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

add_executable(cudart src/main.cu src/camera.h src/hitable.h src/hitable_list.h src/material.h src/cuda_errors.h common/ray.h src/sphere.h common/vec3.h common/options.h)
target_link_libraries(cudart ${MPI_LIBRARIES} ${LibXml2_LIBRARIES})
target_compile_features(cudart PRIVATE cxx_std_11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

# Copy the job scripts
configure_file(job-rt job-rt COPYONLY)
configure_file(job-rt-bench job-rt-bench COPYONLY)
