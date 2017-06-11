cmake_minimum_required(VERSION 3.7.0 FATAL_ERROR)

option (GA_SUPERBUILD "This will build blas+lapack if required" ON)

if(GA_SUPERBUILD)
    project (TAMM_DEPENDENCIES C CXX Fortran)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

    if(DEFINED TAMM_PROC_COUNT)
        set(TAMM_PROC_COUNT ${TAMM_PROC_COUNT})
    else()
        include(ProcessorCount)
        ProcessorCount(TAMM_PROC_COUNT)
        message(STATUS "Number of cores detected = ${TAMM_PROC_COUNT}")
    endif()

    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set (CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/tamm-deps" CACHE PATH "default install path" FORCE)
    endif()

    if (NOT BLAS_LIBRARIES)
        include(blas_lapack)
    else()
        add_custom_target(BLAS_LAPACK ALL)
    endif()

    include(ExternalProject)
    ExternalProject_Add (TAMM_DEPS
            DEPENDS BLAS_LAPACK
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
            BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}
            CMAKE_ARGS -DGA_SUPERBUILD=OFF
            -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
            -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DMPI_INCLUDE_PATH=${MPI_INCLUDE_PATH}
            -DMPI_LIBRARY_PATH=${MPI_LIBRARY_PATH} -DMPI_LIBRARIES=${MPI_LIBRARIES}
            -DBLAS_LIBRARIES=${BLAS_LIBRARIES} -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES}
            -DSCALAPACK_LIBRARIES=${SCALAPACK_LIBRARIES} -DTAMM_PROC_COUNT=${TAMM_PROC_COUNT}
            -DBLAS_INCLUDE_PATH=${BLAS_INCLUDE_PATH} -DBLAS_LIBRARY_PATH=${BLAS_LIBRARY_PATH}
            INSTALL_COMMAND cmake -E echo "Finished installing all TAMM dependencies!"
            )
    return()

else()
#    message(STATUS ${MPI_INCLUDE_PATH})
#    message(STATUS ${MPI_LIBRARY_PATH})
#    message(STATUS ${MPI_LIBRARIES})
#    message(STATUS ${CMAKE_INSTALL_PREFIX})
#    message(STATUS ${TAMM_PROC_COUNT})

    project (NWCHEM C CXX Fortran)

    include(ExternalProject)

    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

    #include(libint)
    #include(eigen)
    include(globalarrays)
    #include(antlr)
endif()
