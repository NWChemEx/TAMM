#
# This file builds Global Arrays via a mock superbuild where the CMakeLists.txt
# in the folder GlobalArrays provides a pretend CMake build for the GA package.
# The present file then establishes the dependencies between GA and BLAS,
# LAPACK, MPI, and SCALAPCK, ensuring the latter are built before trying to
# build the former.  Hence, when the latter calls find_dependency they are
# guaranteed to have been built and can be found.
#
# GlobalArrays includes a CMake build, this is not it.
#
include(UtilityMacros)
include(DependencyMacros)

# GA needs Fortran...ew
enable_language(Fortran)

# Now find or build GA's dependencies
find_or_build_dependency(CBLAS __found_blas)
find_or_build_dependency(LAPACKE __found_lapack)
find_or_build_dependency(NWX_MPI __found_mpi)
#find_or_build_dependency(ScaLAPACK __found_scalapack)

##########################################################
# Determine aggregate remote memory copy interface (ARMCI)
##########################################################

#Possible choices
set(ARMCI_NETWORK_OPTIONS BGML DCMF OPENIB GEMINI DMAPP PORTALS GM VIA
        LAPI MPI-SPAWN MPI-PT MPI-MT MPI-PR MPI-TS MPI3 OFI
        OFA SOCKETS) #MELLANOX

# Get index user choose
is_valid_and_true(ARMCI_NETWORK __set)
if (NOT __set)
    message(STATUS "ARMCI network not set, defaulting to MPI-PR")
    set(ARMCI_NETWORK "--with-mpi-pr")
else()
    list(FIND ARMCI_NETWORK_OPTIONS ${ARMCI_NETWORK} _index)
    if(${_index} EQUAL -1)
        message(WARNING "Unrecognized ARMCI Network, defaulting to MPI-PR")
        set(ARMCI_NETWORK "--with-mpi-pr")
    elseif(${_index} EQUAL 4)
        message(WARNING "We discourage the use of ARMCI_NETWORK=DMAPP")
        message(WARNING "Please consider using ARMCI_NETWORK=MPI-PR instead")
        set(ARMCI_NETWORK "--with-dmapp")
    else()
        string(TOLOWER ${ARMCI_NETWORK} armci_network)
        set(ARMCI_NETWORK "--with-${armci_network}")
    endif()
endif()

message(STATUS ${CMAKE_BINARY_DIR}/stage)
# Add the mock CMake-ified GA project
ExternalProject_Add(GlobalArrays_External
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/GlobalArrays
        CMAKE_ARGS -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
                   -DSUPER_PROJECT_ROOT=${SUPER_PROJECT_ROOT}
                   -DNWX_DEBUG_CMAKE=${NWX_DEBUG_CMAKE}
                   -DARMCI_NETWORK=${ARMCI_NETWORK}
                   -DSTAGE_DIR=${STAGE_DIR}
                   ${CORE_CMAKE_OPTIONS}
        BUILD_ALWAYS 1
        INSTALL_COMMAND ""
        CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                         ${CORE_CMAKE_STRINGS}
)

# Establish the dependencies
add_dependencies(GlobalArrays_External CBLAS_External
                                       LAPACKE_External
                                       NWX_MPI_External
#                                       ScaLAPACK_External
        )


