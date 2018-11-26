# - Try to find Global Arrays
#
#  The user may specify GLOBALARRAYS_ROOT_DIR to aid find_packge in
#  finding an already installed Global Arrays
#
#  Once done this will define
#  GLOBALARRAYS_FOUND - System has Global Arrays
#  GLOBALARRAYS_CONFIG - The ga-config binary path
#  GLOBALARRAYS_INCLUDE_DIR - The Global Arrays include directories
#  GLOBALARRAYS_LIBRARIES - The libraries needed to use Global Arrays

if(NOT DEFINED GLOBALARRAYS_ROOT_DIR)
    find_package(PkgConfig)
    pkg_check_modules(PC_GLOBALARRAYS QUIET ga)
endif()

find_path(GLOBALARRAYS_INCLUDE_DIR ga.h
          HINTS ${PC_GLOBALARRAYS_INCLUDEDIR}
          ${PC_GLOBALARRAYS_INCLUDE_DIRS}
          PATHS ${GLOBALARRAYS_ROOT_DIR}
          )

find_path(GLOBALARRAYS_CONFIG ga-config
          HINTS ${PC_GLOBALARRAYS_BINDIR}
          ${PC_GLOBALARRAYS_BIN_DIRS}
          PATHS ${GLOBALARRAYS_ROOT_DIR}
          PATH_SUFFIXES bin
          )

find_library(GLOBALARRAYS_C_LIBRARY NAMES ga
             HINTS ${PC_GLOBALARRAYS_LIBDIR}
             ${PC_GLOBALARRAYS_LIBRARY_DIRS}
             PATHS ${GLOBALARRAYS_ROOT_DIR}
             )

find_library(GLOBALARRAYS_ARMCI_LIBRARY NAMES armci
             HINTS ${PC_GLOBALARRAYS_LIBDIR}
             ${PC_GLOBALARRAYS_LIBRARY_DIRS}
             PATHS ${GLOBALARRAYS_ROOT_DIR}
             )
find_package_handle_standard_args(GlobalArrays DEFULT_MSG GLOBALARRAYS_C_LIBRARY
                                  GLOBALARRAYS_ARMCI_LIBRARY)

set(GLOBALARRAYS_LIBRARIES ${GLOBALARRAYS_C_LIBRARY}
    ${GLOBALARRAYS_ARMCI_LIBRARY}
    )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GlobalArrays DEFAULT_MSG
                                  GLOBALARRAYS_LIBRARIES
                                  GLOBALARRAYS_INCLUDE_DIR
                                  GLOBALARRAYS_CONFIG
                                  )

set(GLOBALARRAYS_INCLUDE_DIRS ${GLOBALARRAYS_INCLUDE_DIR})
set(GLOBALARRAYS_FOUND ${GlobalArrays_FOUND})

if (GLOBALARRAYS_FOUND)
    #GA, MPI, Blas, Lapack, std fortran libs are already figured out by CMakeBuild
    #Get optional libs using ga-config: pthreads, librt, libm (paths verified by GA)
    execute_process(COMMAND ${GLOBALARRAYS_CONFIG}/ga-config --libs OUTPUT_VARIABLE GA_CONFIG_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${GLOBALARRAYS_CONFIG}/ga-config --flibs OUTPUT_VARIABLE GA_CONFIG_F_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
    #execute_process(COMMAND ${GLOBALARRAYS_CONFIG}/ga-config --ldflags OUTPUT_VARIABLE GA_CONFIG_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

    string(REPLACE " " ";" GA_LIBS_LIST ${GA_CONFIG_LIBS} ${GA_CONFIG_F_LIBS})
    foreach(__lib ${GA_LIBS_LIST})
        if(NOT __lmath)
            string(COMPARE EQUAL "-lm" ${__lib} __lmath)
        endif()
        if(NOT __lpthread)
            string(COMPARE EQUAL "-lpthread" ${__lib} __lpthread)
        endif()
        if(NOT __lrt)
            string(COMPARE EQUAL "-lrt" ${__lib} __lrt)
        endif()
        if(NOT __lgfortran)
            string(COMPARE EQUAL "-lgfortran" ${__lib} __lgfortran)
        endif()
        if(NOT __lifcoremt_pic)
            string(COMPARE EQUAL "-lifcoremt_pic" ${__lib} __lifcoremt_pic)
        endif()
    endforeach()

    if(__lifcoremt_pic)
        enable_language(Fortran)
        find_library(GA_IFCOREMT_LIBRARY
                     NAMES libifcoremt_pic.a libifcoremt_pic.so
                     HINTS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}
                     )
        set(GLOBALARRAYS_LIBRARIES ${GLOBALARRAYS_LIBRARIES} ${GA_IFCOREMT_LIBRARY})
    endif()

    if(__lgfortran)
        enable_language(Fortran)
        find_library(GA_STANDARDFORTRAN_LIBRARY
                     libgfortran${CMAKE_SHARED_LIBRARY_SUFFIX}
                     HINTS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}
                     )
        set(GLOBALARRAYS_LIBRARIES ${GLOBALARRAYS_LIBRARIES} ${GA_STANDARDFORTRAN_LIBRARY})
    endif()

    if(__lpthread)
        set(THREADS_PREFER_PTHREAD_FLAG ON)
        find_package(Threads REQUIRED)
        set(GLOBALARRAYS_LIBRARIES ${GLOBALARRAYS_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
    endif()

    find_path(
        LIBRT_INCLUDE_DIRS
        NAMES time.h
        PATHS ${LIBRT_ROOT_DIR}/include/
    )
    find_library(LIBRT_LIBRARIES rt)

    find_package_handle_standard_args(
        LibRT
        DEFAULT_MSG
        LIBRT_LIBRARIES LIBRT_INCLUDE_DIRS
    )
    if(LIBRT_LIBRARIES)
        set(GLOBALARRAYS_LIBRARIES ${GLOBALARRAYS_LIBRARIES} ${LIBRT_LIBRARIES})
    endif()

    if(__lmath)
        find_path(
            LIBM_INCLUDE_DIRS
            NAMES math.h
            PATHS ${LIBM_ROOT_DIR}/include/
        )

        find_library(LIBM_LIBRARIES m)

        find_package_handle_standard_args(
            LibM
            DEFAULT_MSG
            LIBM_LIBRARIES LIBM_INCLUDE_DIRS
        )
        set(GLOBALARRAYS_LIBRARIES ${GLOBALARRAYS_LIBRARIES} ${LIBM_LIBRARIES})
    endif()
endif()

