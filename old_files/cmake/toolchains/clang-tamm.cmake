
set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_Fortran_COMPILER gfortran)

#------------------ OPTIONAL ---------------------

#Enable/Disable GPU support
set(TAMM_ENABLE_GPU OFF)

# GA CONFIGURATION
#set(ARMCI_NETWORK OPENIB)
#set(USE_OFFLOAD "OFFLOAD" ON)

# BLAS, LAPACK & SCALAPACK. Support only 8-byte integers for now.
#set(BLAS_INCLUDE_PATH /opt/blas/include)
#set(BLAS_LIBRARY_PATH /opt/blas/lib)
# set(BLAS_LIBRARIES "-lblas -llapack" CACHE STRING "BLAS linker flags")
# set(LAPACK_LIBRARIES "${BLAS_LIBRARIES}" CACHE STRING "LAPACK linker flags")
#set(SCALAPACK_LIBRARIES "-mkl -lmkl_scalapack_ilp64 -lmkl_blacs_openmpi_ilp64 -lmkl_intel_thread -lpthread -lm -ldl" CACHE STRING "SCALAPACK linker flags")

# set(LIBINT_ROOT_DIR /opt/tamm-deps) 
# set(ANTLRCPPRUNTIME_ROOT_DIR /opt/tamm-deps) 
# set(EIGEN3_ROOT_DIR /opt/tamm-deps) 
# set(GTEST_ROOT_DIR /opt/tamm-deps) 
# set(BLAS_ROOT_DIR /opt/tamm-deps) 
# set(GLOBALARRAYS_ROOT_DIR /opt/tamm-deps) 

#set(TALSH_ROOT_DIR /opt/TAL_SH)

