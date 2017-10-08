################################################################################
#
# These are macros for finding dependencies.
#
################################################################################

#
# Macro for finding a dependency that is presumed built.  It is assumed that
# whatever name is given
# to this macro, is a name that can be directly fed to CMake's find_package
# macro.  Furthermore it is assumed that the result is given back as the
# standard variables, e.g.
# XXX_INCLUDE_DIRS, XXX_LIBRARIES, etc. where XXX is the name of the library in
# all capital letters.
#
# Syntax: find_dependency(<name> <include> <lib> <flags> <required>)
#    - name     : The name to be passed to find_package
#    - include  : The variable to append the includes on to
#    - lib      : The variable to append the libraries on to
#    - flags    : The variable to append the compile-time flags on to
#    - required : REQUIRED or QUIET will be passed to find_package
#
function(find_dependency __name __include __lib __flags __required)
    find_package(${__name} ${__required})
    string(TOUPPER ${__name} __NAME)
    #Check that variable exists and is set to true
    if(${__NAME}_FOUND AND ${${__NAME}_FOUND})
        if(${__NAME}_INCLUDE_DIRS AND ${${__NAME}_INCLUDE_DIRS})
            list(APPEND ${__include} ${${__NAME}_INCLUDE_DIRS})
            message(STATUS "${__name} Includes: ${${__NAME}_INCLUDE_DIRS}")
        endif()
        if(${__NAME}_LIBRARIES)
            list(APPEND ${__lib} ${${__NAME}_LIBRARIES})
            message(STATUS "${__name} Libs: ${${__NAME}_LIBRARIES}")
        endif()
        if(${__NAME}_DEFINITIONS)
            list(APPEND ${__flags} ${${__NAME}_DEFINITIONS})
            message(STATUS "${__name} Defines: ${${__NAME}_DEFINITIONS}")
        endif()
        set(${__include} ${${__include}} PARENT_SCOPE)
        set(${__lib} ${${__lib}} PARENT_SCOPE)
        set(${__flags} ${${__flags}} PARENT_SCOPE)
    endif()
endfunction()
