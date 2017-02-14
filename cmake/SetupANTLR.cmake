
execute_process(COMMAND mkdir -p "${PROJECT_BINARY_DIR}/downloads/ANTLR/build")
execute_process(COMMAND mkdir -p "${PROJECT_BINARY_DIR}/dependencies/ANTLR/CppRuntime/")

if(NOT $ENV{ANTLR_CPPRUNTIME})

set (ANTLR_CPPRUNTIME $ENV{ANTLR_CPPRUNTIME})

else()

# Build the ANTLR C Runtime library.
include(ExternalProject)
ExternalProject_Add(ANTLR
    PREFIX ANTLR
    URL http://www.antlr.org/download/antlr4-cpp-runtime-4.6-source.zip
    DOWNLOAD_DIR ${PROJECT_BINARY_DIR}/downloads/ANTLR
    SOURCE_DIR ${PROJECT_BINARY_DIR}/downloads/ANTLR/
    INSTALL_DIR ${PROJECT_BINARY_DIR}/dependencies/ANTLR/CppRuntime
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_CXX_COMPILER=g++  -DCMAKE_C_COMPILER=gcc -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    BINARY_DIR ${PROJECT_BINARY_DIR}/downloads/ANTLR/build
    BUILD_COMMAND "make"
    INSTALL_COMMAND make install
)

# Set location of the ANTLR C runtime library.
set (ANTLR_CPPRUNTIME ${PROJECT_BINARY_DIR}/dependencies/ANTLR/CppRuntime)

endif()

# Download the ANTLR binary.
if (NOT EXISTS "${PROJECT_BINARY_DIR}/dependencies/ANTLR/antlr.jar")
    file(DOWNLOAD "http://www.antlr.org/download/antlr-4.6-complete.jar" ${PROJECT_BINARY_DIR}/dependencies/ANTLR/antlr.jar) 
endif()

# Set location of the ANTLR binary.
set(AntlrBinary ${PROJECT_BINARY_DIR}/dependencies/ANTLR/antlr.jar)  


