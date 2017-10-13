if(NOT ARMCI_NETWORK)
    set(GA_ARMCI "--with-mpi-ts") #Default if ARMCI_NETWORK is not set
else()
    string(FIND ${ARMCI_NETWORK} BGML BGML_FOUND)
    string(FIND ${ARMCI_NETWORK} DCMF DCMF_FOUND)
        
    if (${ARMCI_NETWORK} STREQUAL DMAPP)
        message(WARNING "We discourage the use of ARMCI_NETWORK=DMAPP")
        message(WARNING "Please consider using ARMCI_NETWORK=MPI-PR instead")
    endif()

    set(ARMCI_NETWORK_OPTIONS OPENIB GEMINI DMAPP PORTALS GM VIA
        LAPI MPI-SPAWN MPI-PT MPI-MT MPI-PR MPI-TS MPI3 OFI OFA SOCKETS) #MELLANOX

    if(${ARMCI_NETWORK} IN_LIST ARMCI_NETWORK_OPTIONS)
        string(TOLOWER ${ARMCI_NETWORK} armci_network_nwx)
        set(GA_ARMCI "--with-${armci_network_nwx}")
    elseif(${BGML_FOUND} GREATER 0 OR ${ARMCI_NETWORK} STREQUAL BGML)
         set(GA_ARMCI "--with-bgml")
    elseif(${DCMF_FOUND} GREATER 0 OR ${ARMCI_NETWORK} STREQUAL DCMF)
         set(GA_ARMCI "--with-dcmf")
    elseif(${ARMCI_NETWORK} STREQUAL MELLANOX)
         set(GA_ARMCI "--with-openib")
    else()
         message(WARNING "Unknown ARMCI Network: ${ARMCI_NETWORK} provided. Configuring with MPI-TS")
         set(GA_ARMCI "--with-mpi-ts")
    endif()
endif()

#-----------------------------------------------------------------

#TODO: Add an if statement to find_dependency for MPI that will automatically
#      add the libraries to MPI_LIBRARIES and the includes to MPI_INCLUDE_DIRS?
find_package(MPI)
set(_nwx_mpi_libraries ${MPI_C_LIBRARIES} ${MPI_Fortran_LIBRARIES} ${MPI_EXTRA_LIBRARY}) #${MPI_CXX_LIBRARIES}
set(_nwx_mpi_include_dirs ${MPI_C_INCLUDE_PATH} ${MPI_Fortran_INCLUDE_PATH}) #${MPI_CXX_INCLUDE_PATH}

message("_nwx_mpi_include_dirs=${_nwx_mpi_include_dirs}")
message("_nwx_mpi_libraries=${_nwx_mpi_libraries}")

foreach(_nwx_mpi_inc ${_nwx_mpi_include_dirs})
    set(NWX_MPI_INCLUDE_DIRS "${NWX_MPI_INCLUDE_DIRS} -I${_nwx_mpi_inc}")
endforeach()

foreach(_nwx_mpi_lib ${_nwx_mpi_libraries})
    get_filename_component(_nwx_mpi_lib_path ${_nwx_mpi_lib} DIRECTORY) #get lib path
    set(NWX_MPI_LIBRARY_PATH "${NWX_MPI_LIBRARY_PATH} -L${_nwx_mpi_lib_path}")

    get_filename_component(_nwx_mpi_lib_we ${_nwx_mpi_lib} NAME_WE) #get lib name
    string(SUBSTRING ${_nwx_mpi_lib_we} 3 -1 _nwx_mpi_lib_we) #Strip lib prefix from lib name
    set(NWX_MPI_LIBRARIES "${NWX_MPI_LIBRARIES} -l${_nwx_mpi_lib_we}")
endforeach()

set(GA_MPI "--with-mpi=${NWX_MPI_INCLUDE_DIRS} ${NWX_MPI_LIBRARY_PATH} ${NWX_MPI_LIBRARIES}")

#-----------------------------------------------------------

#set(GA_MPI "--with-mpi=-I${MPI_INCLUDE_PATH} -L${MPI_LIBRARY_PATH} ${MPI_LIBRARIES}")

set(GA_SYSVSHMEM "ARMCI_DEFAULT_SHMMAX_UBOUND=131072")

if (USE_OFFLOAD)
    set(GA_OFFLOAD "INTEL_64ALIGN=1")
endif()

if (BLAS_LIBRARIES)
    set(GA_BLAS "--with-blas8=${BLAS_LIBRARIES}")
else()
    #Have cmake build install BLAS+LAPACK and provide it to GA
    set(GA_BLAS "--with-blas8=-lblas -llapack")
endif()

if (LAPACK_LIBRARIES)
    set(GA_LAPACK "--with-lapack=${LAPACK_LIBRARIES}")
else()
    #Have cmake build install BLAS+LAPACK and provide it to GA
    set(GA_LAPACK "--with-lapack=-lblas -llapack")
endif()

if (SCALAPACK_LIBRARIES)
    set(GA_SCALAPACK "--with-scalapack8=${SCALAPACK_LIBRARIES}")
else()
    set(SCALAPACK_LIBRARIES OFF)
    set(GA_SCALAPACK "--without-scalapack")
endif()

# Build GA
ExternalProject_Add(GlobalArrays${TARGET_SUFFIX}
    URL https://github.com/GlobalArrays/ga/releases/download/v5.6.2/ga-5.6.2.tar.gz
    #Pass location where autotools needs to be built 
    CONFIGURE_COMMAND ./autogen.sh 
    COMMAND ./configure --with-tcgmsg 
    ${GA_MPI} --enable-underscoring --disable-mpi-tests #--enable-peigs
    ${GA_SCALAPACK} ${GA_BLAS} ${GA_LAPACK} ${GA_ARMCI} ${GA_OFFLOAD} CC=${CMAKE_C_COMPILER}
    CXX=${CMAKE_CXX_COMPILER} F77=${CMAKE_Fortran_COMPILER} ${GA_SYSVSHMEM} --prefix=${GLOBALARRAYS_ROOT_DIR} #--enable-cxx
    #TODO:Fix LDFLAGS
    LDFLAGS=-L${CMAKE_INSTALL_PREFIX}/blas_lapack/lib
    #BUILD_COMMAND $(MAKE) 
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install
    BUILD_IN_SOURCE 1
    #LOG_CONFIGURE 1
    #LOG_BUILD 1
)
endif()


