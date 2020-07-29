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
             NO_CMAKE_SYSTEM_PATH
        )

find_library(GLOBALARRAYS_ARMCI_LIBRARY NAMES armci
             HINTS ${PC_GLOBALARRAYS_LIBDIR}
                   ${PC_GLOBALARRAYS_LIBRARY_DIRS}
             PATHS ${GLOBALARRAYS_ROOT_DIR}
             NO_CMAKE_SYSTEM_PATH
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

  set(GA_ALL_CONFIG_LIBS "${GA_CONFIG_LIBS} ${GA_CONFIG_F_LIBS}")
  string(REPLACE " " ";" GA_LIBS_LIST ${GA_ALL_CONFIG_LIBS})
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
    if(NOT __lquadmath) 
      string(COMPARE EQUAL "-lquadmath" ${__lib} __lquadmath) 
    endif()        
    if(NOT __lifcoremt_pic) 
      string(COMPARE EQUAL "-lifcoremt_pic" ${__lib} __lifcoremt_pic) 
    endif()  
    if(NOT __libverbs) 
      string(COMPARE EQUAL "-libverbs" ${__lib} __libverbs) 
    endif()        
  endforeach()

  if(__lifcoremt_pic)
    enable_language(Fortran)
    find_library(GA_IFCOREMT_LIBRARY
      NAMES libifcoremt_pic.so libifcoremt_pic.a 
      HINTS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}
    )
    set(GLOBALARRAYS_LIBRARIES ${GLOBALARRAYS_LIBRARIES} ${GA_IFCOREMT_LIBRARY})
  endif()

  if(__lgfortran)
    enable_language(Fortran)
    find_library(GA_STANDARDFORTRAN_LIBRARY
      NAMES libgfortran.so libgfortran.a
      HINTS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}
    )
    set(GLOBALARRAYS_LIBRARIES ${GLOBALARRAYS_LIBRARIES} ${GA_STANDARDFORTRAN_LIBRARY})
  endif()

  if(__lquadmath)
    enable_language(Fortran)
    find_library(GA_QMATH_LIBRARY
      NAMES  libquadmath.so libquadmath.a
      HINTS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}
    )
    set(GLOBALARRAYS_LIBRARIES ${GLOBALARRAYS_LIBRARIES} ${GA_QMATH_LIBRARY})
  endif()

  if(__lpthread)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    set(GLOBALARRAYS_LIBRARIES ${GLOBALARRAYS_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
  endif()

  if(__lrt)
    find_package(LibRT REQUIRED)
  else()
    #For now since GA does not export librt in some configurations
    find_package(LibRT QUIET)
  endif()
  if(LIBRT_LIBRARIES)
    set(GLOBALARRAYS_LIBRARIES ${GLOBALARRAYS_LIBRARIES} ${LIBRT_LIBRARIES})
  endif()

  if(__lmath)
    find_package(LibM REQUIRED)
    set(GLOBALARRAYS_LIBRARIES ${GLOBALARRAYS_LIBRARIES} ${LIBM_LIBRARIES})
  endif()

  if(__libverbs)
    find_package(Ibverbs REQUIRED)
    set(GLOBALARRAYS_LIBRARIES ${GLOBALARRAYS_LIBRARIES} ${IBVERBS_LIBRARIES})
  endif()
endif()



