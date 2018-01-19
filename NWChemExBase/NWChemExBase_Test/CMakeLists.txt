cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(NWChemExBase-TEST VERSION 0.0.0 LANGUAGES CXX)
include(CTest)
enable_testing()

get_filename_component(NWCHEMEXBASE_ROOT ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)

# Tests of CMake Macros
foreach(_macro AssertMacros UtilityMacros)
    file(COPY ${NWCHEMEXBASE_ROOT}/cmake/macros/${_macro}.cmake
         DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY Test${_macro}.cmake DESTINATION ${CMAKE_BINARY_DIR})
    add_test(NAME ${_macro} COMMAND ${CMAKE_COMMAND} -P Test${_macro}.cmake)
endforeach()

#Install test
add_test(NAME InstalledNWXBase COMMAND InstalledNWXBase_files/RunTest.sh
        ${CMAKE_COMMAND})
file(COPY InstalledNWXBase_files DESTINATION ${CMAKE_BINARY_DIR})

# Building Tests
foreach(_test_name DebugHonored ReleaseHonored SimpleRepo SkipTests
                   BuildEigen3 BuildLibInt BuildCBLAS BuildLAPACK
                   BuildBLAS BuildLAPACKE
       )
    add_test(NAME    ${_test_name}
             COMMAND ${CMAKE_BINARY_DIR}/RunNWChemExBaseTest.sh
                     ${_test_name}
                     ${NWCHEMEXBASE_ROOT}
                     "${CMAKE_BINARY_DIR}/${_test_name}_files"
                     ${CMAKE_COMMAND}
            )
    file(COPY ${_test_name}_files DESTINATION ${CMAKE_BINARY_DIR})
endforeach()

file(COPY RunNWChemExBaseTest.sh DESTINATION ${CMAKE_BINARY_DIR})
file(COPY InstalledNWXBase_files DESTINATION ${CMAKE_BINARY_DIR})


