cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TAMM-Python VERSION 0.0.0 LANGUAGES CXX)

# This runs as the TAMM_Python_External sub-build. It consumes the installed
# TAMM exactly like the Tests sub-build does: find_package(TAMM) brings in the
# full transitive dependency closure (includes, defs, link libs), and
# cmsb_add_pymodule builds the pybind11 based python module against it.
find_package(TAMM REQUIRED)
include(TargetMacros)

set(PYTAMM_SRCS     pytamm.cpp)
set(PYTAMM_INCLUDES )
set(PYTAMM_CFLAGS   )
set(PYTAMM_LFLAGS   )

cmsb_add_pymodule(pytamm PYTAMM_SRCS PYTAMM_INCLUDES PYTAMM_CFLAGS PYTAMM_LFLAGS)
