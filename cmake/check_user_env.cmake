if(CMAKE_CXX_COMPILER_ID STREQUAL "XL"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "Cray"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(FATAL_ERROR "TAMM does not support ${CMAKE_CXX_COMPILER_ID} compilers.")
endif()

if("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Darwin")
    if (TAMM_ENABLE_GPU)
        message(FATAL_ERROR "TAMM does not support building with GPU support \
        on MACOSX. Please use TAMM_ENABLE_GPU=OFF for MACOSX builds.")
    endif()
    
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" 
        OR CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
        message(FATAL_ERROR "TAMM does not support ${CMAKE_CXX_COMPILER_ID} compilers on MACOSX.")
    endif()
endif()

macro(get_compiler_exec_name comp_exec_path)
    get_filename_component(comp_exec_name ${comp_exec_path} NAME_WE)
endmacro()

macro(check_compiler_version lang_arg comp_type comp_version)
    if(CMAKE_${lang_arg}_COMPILER_ID STREQUAL "${comp_type}")
        if(CMAKE_${lang_arg}_COMPILER_VERSION VERSION_LESS "${comp_version}")
            get_compiler_exec_name("${CMAKE_${lang_arg}_COMPILER}")
            message(FATAL_ERROR "${comp_exec_name} version provided (${CMAKE_${lang_arg}_COMPILER_VERSION}) \
            is insufficient. Need ${comp_exec_name} >= ${comp_version} for building TAMM.")
        endif()
    endif()
endmacro()

check_compiler_version(C Clang 5)
check_compiler_version(CXX Clang 5)

check_compiler_version(C GNU 7.2)
check_compiler_version(CXX GNU 7.2)
check_compiler_version(Fortran GNU 7.2)

#TODO:Check for GCC>=7 compatibility
# check_compiler_version(C Intel 19)
# check_compiler_version(CXX Intel 19)
# check_compiler_version(Fortran Intel 19)

#TODO:Check for GCC>=7 compatibility
check_compiler_version(C PGI 18)
check_compiler_version(CXX PGI 18)
check_compiler_version(Fortran PGI 18)




