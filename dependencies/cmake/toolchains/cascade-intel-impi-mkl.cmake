
set(CMAKE_BUILD_TYPE Release)

# Compilers (assuming the compilers are in the PATH)
set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpc)
set(CMAKE_Fortran_COMPILER ifort)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#Location where the dependencies need to be installed.
#Default location: cmake_build_folder/tamm-deps
#set(CMAKE_INSTALL_PREFIX /opt/tamm-deps)

#Number of cores to be used for the build
set(TAMM_PROC_COUNT 8)

# GA Options
#set(ARMCI_NETWORK OPENIB)
set(USE_OFFLOAD "OFFLOAD" ON)

# MPI 
set(MPI_INCLUDE_PATH "/msc/apps/compilers/IPS_2017_U3/compilers_and_libraries_2017.3.191/linux/mpi/intel64/include")
set(MPI_LIBRARY_PATH "/msc/apps/compilers/IPS_2017_U3/compilers_and_libraries_2017.3.191/linux/mpi/intel64/lib")
set(MPI_LIBRARIES "-lmpifort -lmpi -lmpigi -ldl -lrt -lpthread")


#set(BLAS_INCLUDE_PATH /opt/intel/mkl/include/)
#set(BLAS_LIBRARY_PATH /opt/intel/mkl/lib/intel64)

# BLAS, LAPACK & SCALAPACK. Support only 8-byte integers for now.
set(BLAS_LIBRARIES "-mkl -lmkl_lapack95_ilp64 -lmkl_blas95_ilp64 -lmkl_core -lmkl_intel_thread -lpthread -lm -ldl" CACHE STRING "BLAS linker flags")
set(LAPACK_LIBRARIES "${BLAS_LIBRARIES}" CACHE STRING "LAPACK linker flags")
set(SCALAPACK_LIBRARIES "-mkl=parallel -lmkl_scalapack_ilp64 -lmkl_blacs_intelmpi_ilp64 -lmkl_intel_thread -lpthread -lm -ldl" 
                        CACHE STRING "SCALAPACK linker flags")


#set(TALSH_INSTALL_PATH /home/gawa722/Cascade/nwchem/TAL_SH)

