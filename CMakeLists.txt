#---------------------------------------------------------------------------------------------------
# CMake Build Configuration for the Ray Tracing Weekend Series
#
# See README.md for guidance.
#---------------------------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.1.0 )

project ( DRT2
  VERSION 3.0.0
  LANGUAGES CXX
)

# Set to c++11
set ( CMAKE_CXX_STANDARD 11 )

# Source
set ( COMMON_ALL
  common/rtweekend.h
  common/camera.h
  common/ray.h
  common/vec3.h
)

set ( SOURCE
  ${COMMON_ALL}
  common/aabb.h
  common/external/stb_image.h
  common/perlin.h
  common/rtw_stb_image.h
  common/texture.h
  src/aarect.h
  src/box.h
  src/bvh.h
  src/hittable.h
  src/hittable_list.h
  src/material.h
  src/onb.h
  src/pdf.h
  src/sphere.h
  src/main.cc
)

# Executables
add_executable(tracer      ${SOURCE})
add_executable(cos_cubed         src/cos_cubed.cc         ${COMMON_ALL})
add_executable(cos_density       src/cos_density.cc       ${COMMON_ALL})
add_executable(integrate_x_sq    src/integrate_x_sq.cc    ${COMMON_ALL})
add_executable(pi                src/pi.cc                ${COMMON_ALL})
add_executable(sphere_importance src/sphere_importance.cc ${COMMON_ALL})
add_executable(sphere_plot       src/sphere_plot.cc       ${COMMON_ALL})

include_directories(common)
