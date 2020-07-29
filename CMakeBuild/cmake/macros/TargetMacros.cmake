################################################################################
#
# Macros for defining new targets.
#
# General definitions so they aren't defined for each function:
#     name      : The name of the target
#     flags     : These are compile-time flags to pass to the compiler
#     includes  : The directories containing include files for the target
#     libraries : The libraries the target needs to link against
#
################################################################################

include(CTest)
enable_testing()

if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        # set(SMS 35 50 52 53 60 61 62 70 72)
        # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
        # foreach(__sm ${SMS})
        #     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${__sm},code=sm_${__sm}")
        # endforeach()
        set(CMAKE_CUDA_FLAGS "--maxrregcount ${CUDA_MAXREGCOUNT} --use_fast_math -gencode arch=compute_${NV_GPU_ARCH},code=sm_${NV_GPU_ARCH} -DUSE_CUDA")
    endif()
endif()


include(UtilityMacros)
include(DependencyMacros)
include(AssertMacros)
include(OptionMacros)

#Little trick so we always know this directory even when we are in a function
set(DIR_OF_TARGET_MACROS ${CMAKE_CURRENT_LIST_DIR})

function(nwchemex_set_up_target __name __flags __lflags __install)
    set(__headers_copy ${${__includes}})
    make_full_paths(__headers_copy)
    foreach(__depend ${NWX_DEPENDENCIES})
        find_dependency(${__depend})
        target_link_libraries(${__name} PRIVATE ${__depend}_External)
        get_property(_tmp_libs TARGET ${__depend}_External
                PROPERTY INTERFACE_LINK_LIBRARIES)
        list(APPEND CMAKE_INSTALL_RPATH ${_tmp_libs})
    endforeach()
    target_link_libraries(${__name} PRIVATE "${_tmp_libs}")
    if(TAMM_EXTRA_LIBS)
        target_link_libraries(${__name} PRIVATE "${TAMM_EXTRA_LIBS}")
    endif()
    target_compile_definitions(${__name} PRIVATE "${__flags}")
    target_include_directories(${__name} PRIVATE ${NWX_INCLUDE_DIR})
    set_property(TARGET ${__name} PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
    set_property(TARGET ${__name} PROPERTY LINK_FLAGS "${__lflags}") 
endfunction()

function(nwchemex_add_executable __name __srcs __flags __lflags)
    set(__srcs_copy ${${__srcs}})
    make_full_paths(__srcs_copy)
    add_executable(${__name} ${__srcs_copy})
    nwchemex_set_up_target(${__name}
                           "${${__flags}}"
                           "${${__lflags}}"
                           bin/${__name})
    install(TARGETS ${__name} DESTINATION bin/${__name})
endfunction()

function(nwchemex_add_library __name __srcs __headers __flags __lflags)
    set(__srcs_copy ${${__srcs}})
    make_full_paths(__srcs_copy)
    is_valid(__srcs_copy HAS_LIBRARY)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    if(HAS_LIBRARY)
        add_library(${__name} ${__srcs_copy})
        nwchemex_set_up_target(${__name}
                "${${_flags}}"
                "${${__lflags}}"
                lib/${__name})
        install(TARGETS ${__name}
                ARCHIVE DESTINATION lib
                LIBRARY DESTINATION lib/${__name}/${__install}
                RUNTIME DESTINATION lib/${__name}/${__install})
    endif()
    set(NWCHEMEX_LIBRARY_NAME ${__name})
    set(NWCHEMEX_LIBRARY_HEADERS ${${__headers}})
    get_filename_component(__CONFIG_FILE ${DIR_OF_TARGET_MACROS} DIRECTORY)
    configure_file("${__CONFIG_FILE}/NWChemExTargetConfig.cmake.in"
                    ${__name}Config.cmake @ONLY)
    install(FILES ${CMAKE_BINARY_DIR}/${__name}Config.cmake
            DESTINATION share/cmake/${__name})
    foreach(__header_i ${${__headers}})
        #We want to preserve structure so get directory (if it exists)
        get_filename_component(__header_i_dir ${__header_i} DIRECTORY)
        install(FILES ${__header_i}
                DESTINATION include/${__name}/${__header_i_dir})
    endforeach()
endfunction()

function(nwchemex_add_pymodule __name __srcs __headers __flags __lflags __init)
    nwchemex_add_library(${__name} ${__srcs} ${__headers} ${__flags}
            ${__lflags})
    SET_TARGET_PROPERTIES(${__name} PROPERTIES PREFIX "")
    install(FILES ${__init}
            DESTINATION lib/${__name})
endfunction()

function(nwchemex_add_test __name __test_file)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(__file_copy ${__test_file})
    make_full_paths(__file_copy)
    add_executable(${__name} ${__file_copy})
    nwchemex_set_up_target(${__name} "" "" "tests")
    install(TARGETS ${__name} DESTINATION tests)
    add_test(NAME ${__name} COMMAND ${CMAKE_BINARY_DIR}/${__name})
    target_include_directories(${__name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    install(FILES ${CMAKE_BINARY_DIR}/CTestTestfile.cmake DESTINATION tests)
endfunction()

function(add_cxx_unit_test __name)
    nwchemex_add_test(${__name} ${__name}.cpp)
    set_tests_properties(${__name} PROPERTIES LABELS "UnitTest")
endfunction()

function(nwchemex_set_up_cuda_target __name __testcudalib __flags __lflags __install)
    set(__headers_copy ${${__includes}})
    make_full_paths(__headers_copy)
    foreach(__depend ${NWX_DEPENDENCIES})
        find_dependency(${__depend})
        target_link_libraries(${__name} PRIVATE ${__depend}_External)
        get_property(_tmp_libs TARGET ${__depend}_External
                PROPERTY INTERFACE_LINK_LIBRARIES)
        list(APPEND CMAKE_INSTALL_RPATH ${_tmp_libs})
    endforeach()

    target_link_libraries(${__name} PRIVATE "${__testcudalib}" "${_tmp_libs}")
    if(TAMM_EXTRA_LIBS)
        target_link_libraries(${__name} PRIVATE "${TAMM_EXTRA_LIBS}")
    endif()
    target_compile_definitions(${__name} PRIVATE "${__flags}")
    target_include_directories(${__name} PRIVATE ${NWX_INCLUDE_DIR} ${CUDA_TOOLKIT_INCLUDE})
    set_property(TARGET ${__name} PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
    set_property(TARGET ${__name} PROPERTY LINK_FLAGS "${__lflags}") 
    set_property(TARGET ${__name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endfunction()

function(add_mpi_cuda_unit_test __name __cudasrcs __np __testargs)
    string(TOUPPER ${__name} __NAME)
    string(SUBSTRING ${__NAME} 0 5 _test_prefix)
    set(_dest_install_folder "methods")
    if(_test_prefix STREQUAL "TEST_")
        set(_dest_install_folder "tests")
    endif()

    set(__test_file ${__name}.cpp)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(__file_copy ${__test_file})
    make_full_paths(__file_copy)
    
    #set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    foreach(__depend ${NWX_DEPENDENCIES})
        find_dependency(${__depend})
        get_property(_tmp_incs TARGET ${__depend}_External
                PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
        list(APPEND nwx_cuda_incs ${_tmp_incs})
    endforeach()

    set(__testcudalib "tce_${__name}")

    add_library(${__testcudalib} ${__cudasrcs})
    target_include_directories(${__testcudalib} PRIVATE ${nwx_cuda_incs} ${CUDA_TOOLKIT_INCLUDE})
    set_property(TARGET ${__testcudalib} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    
    add_executable(${__name} ${__file_copy})
    #set_property(TARGET ${__name} PROPERTY LINKER_LANGUAGE CXX)
    #set_source_files_properties(${sources} PROPERTIES LANGUAGE "CUDA")

    nwchemex_set_up_cuda_target(${__name} ${__testcudalib} "" "" ${_dest_install_folder})
    install(TARGETS ${__name} DESTINATION ${_dest_install_folder})
    set(__nwx_mpiexec ${MPIEXEC_EXECUTABLE})
    if(MPIEXEC)
      set(__nwx_mpiexec ${MPIEXEC})
    endif()
    add_test(NAME ${__name} COMMAND ${__nwx_mpiexec} ${MPIEXEC_NUMPROC_FLAG} ${__np} ${CMAKE_BINARY_DIR}/${__name} ${__testargs})
    target_include_directories(${__name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    install(FILES ${CMAKE_BINARY_DIR}/CTestTestfile.cmake DESTINATION ${_dest_install_folder})
    set_tests_properties(${__name} PROPERTIES LABELS "MPICUDAUnitTest")
endfunction()

function(add_mpi_unit_test __name __np __testargs)
    string(TOUPPER ${__name} __NAME)
    string(SUBSTRING ${__NAME} 0 5 _test_prefix)
    set(_dest_install_folder "methods")
    if(_test_prefix STREQUAL "TEST_")
        set(_dest_install_folder "tests")
    endif()

    set(__test_file ${__name}.cpp)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(__file_copy ${__test_file})
    make_full_paths(__file_copy)
    add_executable(${__name} ${__file_copy})
    nwchemex_set_up_target(${__name} "" "" ${_dest_install_folder})
    install(TARGETS ${__name} DESTINATION ${_dest_install_folder})
    set(__nwx_mpiexec ${MPIEXEC_EXECUTABLE})
    if(MPIEXEC)
      set(__nwx_mpiexec ${MPIEXEC})
    endif()
    add_test(NAME ${__name} COMMAND ${__nwx_mpiexec} ${MPIEXEC_NUMPROC_FLAG} ${__np} ${CMAKE_BINARY_DIR}/${__name} ${__testargs})
    target_include_directories(${__name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    install(FILES ${CMAKE_BINARY_DIR}/CTestTestfile.cmake DESTINATION ${_dest_install_folder})
    set_tests_properties(${__name} PROPERTIES LABELS "MPIUnitTest")
endfunction()

function(add_python_unit_test __name)
    foreach(__depend ${NWX_DEPENDENCIES})
        find_dependency(${__depend})
        get_property(_tmp_libs TARGET ${__depend}_External
                PROPERTY INTERFACE_LINK_LIBRARIES)
        foreach(__lib ${_tmp_libs})
            get_filename_component(_lib_path ${__lib} DIRECTORY)
            set(LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:${_lib_path}")
        endforeach()
    endforeach()
    set(env_vars)
    add_test(NAME Py${__name} COMMAND python3 ${__name}.py)
    set_tests_properties(Py${__name} PROPERTIES ENVIRONMENT
       "LD_LIBRARY_PATH=${LD_LIBRARY_PATH};PYTHONPATH=${STAGE_INSTALL_DIR}/lib")
    install(FILES ${__name}.py DESTINATION tests)
endfunction()

function(add_cmake_macro_test __name)
    install(FILES ${__name}.cmake DESTINATION tests)
    list(GET CMAKE_MODULE_PATH 0 _stage_dir)
    set(_macro_dir "${_stage_dir}/share/cmake/CMakeBuild/macros")
    add_test(NAME ${__name}
             COMMAND ${CMAKE_COMMAND} -DCMAKE_MODULE_PATH=${_macro_dir}
                                      -P ${__name}.cmake)
    install(FILES ${CMAKE_BINARY_DIR}/CTestTestfile.cmake DESTINATION tests)
endfunction()

function(add_nwxbase_test __name)
    include(ExternalProject)
    bundle_cmake_args(CMAKE_CORE_OPTIONS CMAKE_CXX_COMPILER CMAKE_C_COMPILER
            CMAKE_Fortran_COMPILER CMAKE_INSTALL_PREFIX)

    ExternalProject_Add(${__name}
        PREFIX ${__name}
        DOWNLOAD_DIR ${__name}
        BINARY_DIR ${__name}/build
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/${__name}
        CMAKE_ARGS ${CMAKE_CORE_OPTIONS}
        BUILD_ALWAYS 1
        CMAKE_CACHE_ARGS -DCMAKE_PREFIX_PATH:LIST=${CMAKE_PREFIX_PATH}
                         -DCMAKE_MODULE_PATH:LIST=${CMAKE_MODULE_PATH}
        INSTALL_COMMAND ""
    )
    set(working_dir ${CMAKE_BINARY_DIR}/${__name}/build/test_stage/${CMAKE_INSTALL_PREFIX})
    add_test(NAME ${__name}
             COMMAND ${working_dir}/tests/${__name})
    install(FILES ${CMAKE_BINARY_DIR}/CTestTestfile.cmake DESTINATION tests)
endfunction()
