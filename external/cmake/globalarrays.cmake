
# Set GA install path .
set (GA_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/ga)

if(EXISTS ${CMAKE_INSTALL_PREFIX}/ga/lib/libga.a)
    add_custom_target(GLOBALARRAYS ALL)
else()
    message("Building Global Arrays 5.6.2")
    set(GA_VERSION ga-5.6.2)


    if(NOT ARMCI_NETWORK)
        set(GA_ARMCI "--with-mpi-ts") #Default if ARMCI_NETWORK is not set
    else()
        string(FIND ${ARMCI_NETWORK} "BGML" BGML_FOUND)
        string(FIND ${ARMCI_NETWORK} "DCMF" DCMF_FOUND)

        if (${ARMCI_NETWORK} STREQUAL OPENIB)
            set(GA_ARMCI "--with-openib")
        elseif(${BGML_FOUND} GREATER 0)
            set(GA_ARMCI "--with-bgml")
        elseif(${DCMF_FOUND} GREATER 0)
            set(GA_ARMCI "--with-dcmf") 
        elseif (${ARMCI_NETWORK} STREQUAL GEMINI)   
            set(GA_ARMCI "--with-gemini") 
        elseif (${ARMCI_NETWORK} STREQUAL DMAPP)   
            message(WARNING "We discourage the use of ARMCI_NETWORK=DMAPP")
            message(WARNING "Please ARMCI_NETWORK=MPI-PR instead")
            set(GA_ARMCI "--with-dmapp")     
        elseif (${ARMCI_NETWORK} STREQUAL PORTALS)   
            set(GA_ARMCI "--with-portals") 
        elseif (${ARMCI_NETWORK} STREQUAL GM)   
            set(GA_ARMCI "--with-gm")       
        elseif (${ARMCI_NETWORK} STREQUAL VIA)   
            set(GA_ARMCI "--with-via") 
        elseif (${ARMCI_NETWORK} STREQUAL MELLANOX)
            set(GA_ARMCI "--with-openib")    
        elseif (${ARMCI_NETWORK} STREQUAL LAPI)
            set(GA_ARMCI "--with-lapi")    
        elseif (${ARMCI_NETWORK} STREQUAL MPI-SPAWN)
            set(GA_ARMCI "--with-mpi-spawn")    
        elseif (${ARMCI_NETWORK} STREQUAL MPI-PT)
            set(GA_ARMCI "--with-mpi-pt")   
        elseif (${ARMCI_NETWORK} STREQUAL MPI-MT)
            set(GA_ARMCI "--with-mpi-mt")   
        elseif (${ARMCI_NETWORK} STREQUAL MPI-PR)
            set(GA_ARMCI "--with-mpi-pr")   
        elseif (${ARMCI_NETWORK} STREQUAL MPI-TS)
            set(GA_ARMCI "--with-mpi-ts")   
        elseif (${ARMCI_NETWORK} STREQUAL MPI3)
            set(GA_ARMCI "--with-mpi3")      
        elseif (${ARMCI_NETWORK} STREQUAL OFI)
            set(GA_ARMCI "--with-ofi")   
        elseif (${ARMCI_NETWORK} STREQUAL OFA)
            set(GA_ARMCI "--with-ofa")   
        elseif (${ARMCI_NETWORK} STREQUAL SOCKETS)
            set(GA_ARMCI "--with-sockets")               
        #elseif (${ARMCI_NETWORK} STREQUAL ARMCI)      
        #   set(GA_ARMCI "--with-armci")    
        else()
            message(WARNING "Unknown ARMCI Network ${ARMCI_NETWORK} provided. Using MPI-TS")
            set(GA_ARMCI "--with-mpi-ts")
        endif()
endif()

set(GA_MPI "--with-mpi=-I${MPI_INCLUDE_PATH} -L${MPI_LIBRARY_PATH} ${MPI_LIBRARIES}")

set(GA_SYSVSHMEM "ARMCI_DEFAULT_SHMMAX_UBOUND=131072")

if (USE_OFFLOAD)
    set(GA_OFFLOAD "INTEL_64ALIGN=1")
endif()



if (DEFINED BLAS_LIBRARIES)
    set(GA_BLAS "--with-blas8=${BLAS_LIBRARIES}")
else()
    #Assume scalapack is not provided if blas is not specified  
    set(GA_BLAS "--with-blas8=-lblas -llapack --without-scalapack")
endif()

if (DEFINED LAPACK_LIBRARIES)
    set(GA_LAPACK "--with-lapack=-lblas -llapack")
endif()

if (DEFINED SCALAPACK_LIBRARIES)
    set(GA_SCALAPACK "--with-scalapack8=${SCALAPACK_LIBRARIES}")
else()
    set(SCALAPACK_LIBRARIES OFF)
endif()


# Build GA
include(ExternalProject)
ExternalProject_Add(GLOBALARRAYS
    PREFIX GLOBALARRAYS
    URL https://github.com/GlobalArrays/ga/releases/download/v5.6.2/ga-5.6.2.tar.gz
    # GIT_REPOSITORY https://github.com/GlobalArrays/ga.git
    # GIT_TAG "hotfix/5.6.1"
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/${GA_VERSION}
    #Pass location where autotools needs to be built 
    CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/external/${GA_VERSION}/autogen.sh 
     #${CMAKE_CURRENT_BINARY_DIR}/external/${GA_VERSION}/autotools 
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/external/${GA_VERSION}/configure --with-tcgmsg 
    ${GA_MPI} --enable-underscoring --disable-mpi-tests #--enable-peigs
    ${GA_SCALAPACK} ${GA_BLAS} ${GA_LAPACK} ${GA_ARMCI} ${GA_OFFLOAD} CC=${CMAKE_C_COMPILER}
    CXX=${CMAKE_CXX_COMPILER} F77=${CMAKE_Fortran_COMPILER} ${GA_SYSVSHMEM} --prefix=${GA_INSTALL_PATH} #--enable-cxx
    LDFLAGS=-L${CMAKE_INSTALL_PREFIX}/blas_lapack/lib
    BUILD_COMMAND make -j${TAMM_PROC_COUNT}
    INSTALL_COMMAND make install
    BUILD_IN_SOURCE 1
    #LOG_CONFIGURE 1
    #LOG_BUILD 1
)
endif()


