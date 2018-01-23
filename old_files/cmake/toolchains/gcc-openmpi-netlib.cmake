
set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
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

#Location where the dependencies need to be installed.
#Default location: cmake_build_folder/tamm-deps
#set(CMAKE_INSTALL_PREFIX /opt/tamm-deps)

#set(LIBINT_INSTALL_PATH /opt/libint2) 
#set(ANTLR_CPPRUNTIME_PATH /opt/ANTLR/CppRuntime) 
#set(EIGEN3_INSTALL_PATH /opt/eigen3) 
#set(GTEST_INSTALL_PATH /opt/googletest) 

#set(TALSH_INSTALL_PATH /opt/TAL_SH)

