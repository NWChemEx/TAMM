# ------------------------------------------------------------------------------
#  Copyright (C) 2016-2017, Pacific Northwest National Laboratory
#  This software is subject to copyright protection under the laws of the
#  United States and other countries
#
#  All rights in this computer software are reserved by the
#  Pacific Northwest National Laboratory (PNNL)
#  Operated by Battelle for the U.S. Department of Energy
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
project (TAMM_LIBRARY C CXX)
#enable_language (Fortran)

set(CMAKE_BUILD_TYPE Debug)

# The version number.
set (TAMM_VERSION_MAJOR 0)
set (TAMM_VERSION_MINOR 1)

#Cannot set the following since cmake cannot handle clang
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

INCLUDE(CTest)
# Should a test fail, display the test program output
set (CTEST_OUTPUT_ON_FAILURE TRUE)

if(TAMM_PROC_COUNT)
    set(TAMM_PROC_COUNT ${TAMM_PROC_COUNT})
else()
    include(ProcessorCount)
    ProcessorCount(TAMM_PROC_COUNT)
    message(STATUS "Number of cores detected = ${TAMM_PROC_COUNT}")
endif()
message(STATUS "Using ${TAMM_PROC_COUNT} cores for building TAMM dependencies and running tests.")
message("Please set the TAMM_PROC_COUNT cmake variable to change the cpu count.")

if(EIGEN3_INSTALL_PATH)
    message(STATUS "Found eigen3: ${EIGEN3_INSTALL_PATH}")
else()
    message(FATAL_ERROR "Eigen3 not found. Please install TAMM dependencies first")
endif()

include_directories( ${PROJECT_SOURCE_DIR} )

set(tammxSources
        tammx/tammx.cc
        tammx/index_sort.cc)

add_library(tammx STATIC ${tammxSources})
target_include_directories(tammx SYSTEM PUBLIC ${PROJECT_BINARY_DIR} ${EIGEN3_INSTALL_PATH}/include/eigen3)

add_executable(tammx_main tammx/tammx_main.cc)
target_link_libraries(tammx_main tammx)
#set_property(TARGET tammx_main PROPERTY CXX_STANDARD 14)
#set_property(TARGET tammx_main PROPERTY CXX_STANDARD_REQUIRED ON)

add_executable(ccsd_driver tammx/ccsd_driver.cc nwchem_main/hartree_fock.cc)
add_dependencies(ccsd_driver tammx)
target_include_directories(ccsd_driver SYSTEM PUBLIC ${PROJECT_SOURCE_DIR} ${LIBINT_INSTALL_PATH}/include ${EIGEN3_INSTALL_PATH}/include/eigen3)
target_link_libraries(ccsd_driver tammx -L${LIBINT_INSTALL_PATH}/lib -lint2 -lrt)

# add_executable(hf nwchem_main/hartree_fock.cc)
# target_include_directories(hf SYSTEM PUBLIC ${LIBINT_INSTALL_PATH}/include ${EIGEN3_INSTALL_PATH}/include/eigen3)
# target_link_libraries(hf -L${LIBINT_INSTALL_PATH}/lib -lint2)

add_executable(tammx_talsh EXCLUDE_FROM_ALL tammx/tammx_talsh.cc tammx/tammx.cc tammx/index_sort.cc)
target_include_directories(tammx_talsh SYSTEM PUBLIC  ${PROJECT_BINARY_DIR}
${TALSH_INSTALL_PATH})
target_link_libraries(tammx_talsh -L${TALSH_INSTALL_PATH} -ltalsh)


# Install
if(NOT TAMM_INSTALL_PREFIX)
    set(TAMM_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/tamm-install)
endif()

install(FILES ${PROJECT_SOURCE_DIR}/nwchem_main/h2o.xyz DESTINATION ${PROJECT_BINARY_DIR})

# Find Doxygen
FIND_PACKAGE(Doxygen)

if(DOXYGEN_FOUND)
    set(DOCS "${PROJECT_BINARY_DIR}/docs")
    message(STATUS "Documentation will be built at: " ${DOCS})
    execute_process(COMMAND mkdir -p ${DOCS})
    set(DOCOUTPUT ${DOCS}/html)
    # Custom Target to build the Doxygen documentation.
    ADD_CUSTOM_COMMAND(OUTPUT ${DOCOUTPUT}
            COMMAND cmake -E copy ${PROJECT_SOURCE_DIR}/Doxyfile.in ${DOCS}/Doxyfile.in
            COMMAND cmake -E copy_directory ${PROJECT_SOURCE_DIR}/tammx ${DOCS}/tammx
#            COMMAND cmake -E copy_directory ${PROJECT_SOURCE_DIR}/tensor ${DOCS}/tensor
            # Change to docs folder before building the documentation.
            COMMAND cmake -E chdir ${DOCS} doxygen ${DOCS}/Doxyfile.in
#            COMMAND cmake -E remove_directory ${DOCS}/tamm_parser
            COMMAND cmake -E remove_directory ${DOCS}/tammx
            COMMAND cmake -E remove -f ${DOCS}/Doxyfile.in
            DEPENDS ${PROJECT_SOURCE_DIR}/Doxyfile.in
            COMMENT "Building Doxygen documentation."
            VERBATIM)
    ADD_CUSTOM_TARGET(docs DEPENDS ${DOCOUTPUT})
endif()



##UNCOMMENT-OLD TAMM
#Uncomment if we use additional cmake modules later
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

#find_path (BLAS_INCLUDE_PATH NAMES cblas.h mkl_cblas.h PATHS ${BLAS_INCLUDE_PATH})
#message(STATUS "BLAS headers found at: ${BLAS_INCLUDE_PATH}")

##Load the GA_CONFIG variable
#if(NOT GA_CONFIG)
#    message(FATAL_ERROR "TAMM ERROR: CMAKE variable GA_CONFIG not set")
#else()
#    set(GA_CONFIG ${GA_CONFIG}/ga-config)
#endif()
#
#if(NOT NWCHEM_TOP)
#    message(FATAL_ERROR "TAMM ERROR: CMAKE variable NWCHEM_TOP not set")
#else()
#    message(STATUS "NWCHEM_TOP set to: ${NWCHEM_TOP}")
#endif()

#add_definitions(-DMPICH_IGNORE_CXX_SEEK)
#execute_process(COMMAND ${GA_CONFIG} --cc OUTPUT_VARIABLE tamm_c_compiler OUTPUT_STRIP_TRAILING_WHITESPACE)
#execute_process(COMMAND ${GA_CONFIG} --f77 OUTPUT_VARIABLE tamm_f77_compiler OUTPUT_STRIP_TRAILING_WHITESPACE)
#
#SET(CMAKE_C_COMPILER ${tamm_c_compiler})
#set(CMAKE_Fortran_COMPILER ${tamm_f77_compiler})

#Assume Cmake picks the right c++ compiler for now
#SET(CMAKE_CXX_COMPILER mpicxx)

#execute_process(COMMAND ${GA_CONFIG} --cppflags --cflags OUTPUT_VARIABLE tamm_cflags OUTPUT_STRIP_TRAILING_WHITESPACE)
#execute_process(COMMAND ${GA_CONFIG} --ldflags OUTPUT_VARIABLE tamm_ldflags OUTPUT_STRIP_TRAILING_WHITESPACE)
#execute_process(COMMAND ${GA_CONFIG} --libs OUTPUT_VARIABLE tamm_libs OUTPUT_STRIP_TRAILING_WHITESPACE)
#
#set(ParserSrcDir ${PROJECT_SOURCE_DIR}/frontend)
#
## Collect all the parser sources.
#set(tammParserSources
#    frontend/Parse.cc
#    frontend/Error.cc
#    frontend/TAMMLexer.cc
#    frontend/TAMMParser.cc
#    frontend/ASTBuilder.cc
#    frontend/TAMMVisitor.cc
#    frontend/Util.cc
#    frontend/Semant.cc
#    frontend/Intermediate.cc
#    )
#
#set(tammLibSources
#  tensor/capi.cc
#  tensor/corf.cc
#  tensor/dummy.cc
#  tensor/equations.cc
#  tensor/expression.cc
#  tensor/func.cc
#  tensor/gmem.cc
#  tensor/index_sort.cc
#  tensor/preprocess.cc
#  tensor/schedulers.cc
#  tensor/stats.cc
#  tensor/t_assign.cc
#  tensor/t_mult.cc
#  tensor/tensor.cc
#  tensor/tensors_and_ops.cc
#  tensor/variables.cc
#  #noga_HF/noga_fock_build_ref.cc
#  )
#
#set(refCodeSources
#  reference_code/cisd_e_ref.cc
#  reference_code/cisd_c1_ref.cc
#  reference_code/cisd_c2_ref.cc
#  reference_code/cc2_t1_ref.cc
#  reference_code/cc2_t2_ref.cc
#  reference_code/ccsd_t12.cc
#  reference_code/ccsd_lambda1_ref.cc
#  reference_code/ccsd_lambda2_ref.cc
#  reference_code/icsd_t1_ref.cc
#  reference_code/ccsd_e_ref.cc
#  reference_code/ccsd_t1_ref.cc
#  reference_code/ccsd_t2_ref.cc
#  reference_code/icsd_t2_ref.cc
#  reference_code/ipccsd_x1_ref.cc
#  reference_code/ipccsd_x2_ref.cc
#  reference_code/eaccsd_x1_ref.cc
#  reference_code/eaccsd_x2_ref.cc
#  reference_code/ccsd_1prdm_hh_ref.cc
#  reference_code/ccsd_1prdm_pp_ref.cc
#  reference_code/ccsd_1prdm_hp_ref.cc
#  reference_code/ccsd_1prdm_ph_ref.cc
#)
#
#set(tammLibSources
#        tensor/capi.cc
#        tensor/corf.cc
#        tensor/dummy.cc
#        tensor/equations.cc
#        tensor/expression.cc
#        tensor/func.cc
#        tensor/gmem.cc
#        tensor/index_sort.cc
#        tensor/preprocess.cc
#        tensor/schedulers.cc
#        tensor/stats.cc
#        tensor/t_assign.cc
#        tensor/t_mult.cc
#        tensor/tensor.cc
#        tensor/tensors_and_ops.cc
#        tensor/variables.cc
#        )

#
#string(REPLACE "-I" "" tamm_cmake_cflags ${tamm_cflags})
##To be recognized as a list, the strings must be seperated by ;
#string(REPLACE " " ";" tamm_cflags ${tamm_cmake_cflags})
#
#message(STATUS "ANTLR C++ runtime found at: " ${ANTLR_CPPRUNTIME_PATH})
#ADD_LIBRARY(a4runtime STATIC IMPORTED)
#SET_TARGET_PROPERTIES(a4runtime PROPERTIES IMPORTED_LOCATION ${ANTLR_CPPRUNTIME_PATH}/lib/libantlr4-runtime.a)
#
##SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEXT_INT -DGFORTRAN")
#add_library(tamm EXCLUDE_FROM_ALL STATIC ${tammParserSources} ${tammLibSources} ${refCodeSources})
#link_directories(${ANTLR_CPPRUNTIME_PATH}/lib ${TAMM_BLAS_LIB})
#target_include_directories(tamm SYSTEM PUBLIC  ${PROJECT_BINARY_DIR} ${ParserSrcDir} ${tamm_cflags}
#                        ${BLAS_INCLUDE_PATH} ${ANTLR_CPPRUNTIME_PATH}/include/antlr4-runtime/)
#target_link_libraries(tamm a4runtime)
#
##set_property(TARGET tamm PROPERTY CXX_STANDARD 14)
##set_property(TARGET tamm PROPERTY CXX_STANDARD_REQUIRED ON)
#
## Build the tamm_parser_test code
#add_executable(tamm_parser_test ${ParserSrcDir}/test.cc)
#add_dependencies(tamm_parser_test tamm)
#target_include_directories(tamm_parser_test SYSTEM PUBLIC  ${ParserSrcDir})
## Set target libraries to be linked.
#target_link_libraries(tamm_parser_test tamm a4runtime)
#
#find_path(cblas_header_name "mkl_cblas.h" ${BLAS_INCLUDE_PATH})
#if (NOT cblas_header_name)
#    set(cblas_header_name "cblas.h")
#else()
#    set(cblas_header_name "mkl_cblas.h")
#endif()
##message(STATUS "cblas header: " ${cblas_header_name})


#add_executable(tamm_main tensor/main.cc tensor/print.cc) #EXCLUDE_FROM_ALL
#add_dependencies(tamm_main tamm)
#target_include_directories(tamm_main SYSTEM PUBLIC  ${PROJECT_BINARY_DIR} ${ParserSrcDir} ${tamm_cflags} ${BLAS_INCLUDE_PATH})
## Set target libraries to be linked.
#target_link_libraries(tamm_main tamm ${tamm_ldflags} ${BLAS_LIBRARIES}) #${tamm_libs}


#UNCOMMENT
#message(STATUS "Installing TAMM library at: " ${TAMM_INSTALL_PREFIX})
#
#install(TARGETS tamm
#        DESTINATION ${TAMM_INSTALL_PREFIX}/lib)
#
#file(GLOB TAMM_HEADERS RELATIVE ${PROJECT_SOURCE_DIR} "${PROJECT_SOURCE_DIR}/frontend/*.h" "${PROJECT_SOURCE_DIR}/tensor/*.h")
#install(FILES ${TAMM_HEADERS} DESTINATION ${TAMM_INSTALL_PREFIX}/include)
#
#file(GLOB EQFILES RELATIVE ${PROJECT_SOURCE_DIR}  "${PROJECT_SOURCE_DIR}/tamm_tests/NWChemEqs/*.eq")
#install(FILES ${EQFILES} DESTINATION ${TAMM_INSTALL_PREFIX}/tests/NWChemEqs)

## Create directory to store test results
#execute_process(COMMAND mkdir -p "TestResults/ParserTests")
#
## Run each test from the parser testsuite
#foreach (test ${EQFILES})
#    get_filename_component(Testname ${test} NAME_WE)
#    add_test(${test} python ${PROJECT_SOURCE_DIR}/scripts/testing/ParserTest.py "${Testname}"
#            "${PROJECT_BINARY_DIR}/tamm_parser_test" "${PROJECT_SOURCE_DIR}/tamm_tests/NWChemEqs/"
#            "${PROJECT_BINARY_DIR}/TestResults/ParserTests")
#endforeach ()


#get_filename_component(NW_CC ${CMAKE_C_COMPILER} NAME )
#get_filename_component(NW_FC ${CMAKE_Fortran_COMPILER} NAME )
#get_filename_component(NW_AR ${CMAKE_AR} NAME )
#
#if (NWCHEM_CMAKE_BUILD)
#    set(patch_script ./nwchem-cmake-fix.bash)
#    set(unpatch_script ./nwchem-cmake-unpatch.bash)
#    set(make_link_cmd make)
#    set(NWCHEM_BINARY ${NWCHEM_BUILD_DIR}/nwchem)
#else()
#    set(patch_script ./nwchem-fix.bash)
#    set(unpatch_script ./nwchem-unpatch.bash)
#    set(make_link_cmd make CC=${NW_CC} FC=${NW_FC} AR=${NW_AR} link)
#    set(NWCHEM_BINARY ${NWCHEM_TOP}/bin/LINUX64/nwchem)
#endif()
#
#add_custom_target("style")
#add_custom_command(TARGET "style" PRE_BUILD
#  COMMAND ${CMAKE_COMMAND} ARGS -E chdir ${CMAKE_SOURCE_DIR}/scripts ./cpplint.bash "${CMAKE_SOURCE_DIR}/tensor/*"
#  COMMENT "=================== Checking style..." )
#
#
#add_custom_target("patch")
#add_custom_command(TARGET "patch" POST_BUILD
#    COMMAND ${CMAKE_COMMAND} ARGS -E chdir ${CMAKE_SOURCE_DIR}/nwchem-mods ${patch_script}
#     ${NWCHEM_TOP} ${TAMM_INSTALL_PREFIX}/lib ${ANTLR_CPPRUNTIME_PATH}/lib/libantlr4-runtime.a
#    COMMENT "=================== Patching nwchem..." )
#
#add_custom_target("unpatch")
#add_custom_command(TARGET "unpatch" POST_BUILD
#    COMMAND ${CMAKE_COMMAND} ARGS -E chdir ${CMAKE_SOURCE_DIR}/nwchem-mods ${unpatch_script} ${NWCHEM_TOP}
#    COMMENT "=================== Unpatching nwchem GNUMakefile/CMakeLists ..." )
#
#add_custom_target(link COMMAND "${CMAKE_COMMAND}" --build . --target install)
#
#
#
#add_custom_command(TARGET "link" POST_BUILD
#    COMMAND ${CMAKE_COMMAND} ARGS -E chdir ${NWCHEM_BUILD_DIR} ${make_link_cmd}
#    COMMENT "=================== Relinking nwchem..." )
#
#set (TEST_COMMAND "mpirun -n ${TAMM_PROC_COUNT} ${NWCHEM_BINARY}")
#
#add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose)
#
#file(GLOB INTEG_TESTS RELATIVE ${PROJECT_SOURCE_DIR} "${PROJECT_SOURCE_DIR}/CI_testing/integration/inputs/*.nw")
#foreach (test ${INTEG_TESTS})
#    get_filename_component(Testname ${test} NAME_WE)
#    #string(REGEX REPLACE "\\.[^.]*$" "" Testname "${Testname}")
#    add_test(${test} python ${PROJECT_SOURCE_DIR}/scripts/testing/CheckResults.py "${NWCHEM_BINARY}" "${Testname}" "${TAMM_PROC_COUNT}")
#endforeach()
#
#SET(TAMM_EQ_PATH ${TAMM_INSTALL_PREFIX}/tests/NWChemEqs)
#add_definitions(-DTAMM_EQ_PATH=\"${TAMM_EQ_PATH}\")
