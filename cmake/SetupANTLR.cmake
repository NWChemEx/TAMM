

if(NOT ${ANTLR_CPPRUNTIME})

set (ANTLR_CPPRUNTIME_DIR ${ANTLR_CPPRUNTIME})

else()

execute_process(COMMAND mkdir -p "${PROJECT_BINARY_DIR}/dependencies/ANTLR/buildCppRuntime/")
execute_process(COMMAND mkdir -p "${PROJECT_BINARY_DIR}/dependencies/ANTLR/CppRuntime/")
file(COPY ${PROJECT_SOURCE_DIR}/dependencies/ANTLR4/ DESTINATION ${PROJECT_BINARY_DIR}/dependencies/ANTLR/buildCppRuntime/)

# Build the ANTLR C Runtime library.
include(ExternalProject)
ExternalProject_Add(ANTLR
    PREFIX ANTLR
    SOURCE_DIR ${PROJECT_BINARY_DIR}/dependencies/ANTLR/buildCppRuntime/antlr4-cpp-runtime
    INSTALL_DIR ${PROJECT_BINARY_DIR}/dependencies/ANTLR/CppRuntime
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    BINARY_DIR ${PROJECT_BINARY_DIR}/dependencies/ANTLR/buildCppRuntime/antlr4-cpp-runtime/build
    BUILD_COMMAND make -j8
    INSTALL_COMMAND make install
)

# Set location of the ANTLR C runtime library.
set (ANTLR_CPPRUNTIME_DIR ${PROJECT_BINARY_DIR}/dependencies/ANTLR/CppRuntime)

endif()

file(COPY ${PROJECT_SOURCE_DIR}/dependencies/ANTLR4/antlr-4.6-complete.jar DESTINATION ${PROJECT_BINARY_DIR}/dependencies/ANTLR/)
# Set location of the ANTLR binary.
set(AntlrBinary ${PROJECT_BINARY_DIR}/dependencies/ANTLR/antlr-4.6-complete.jar)  


