################################################################################
#                                                                              #
# These are macros useful for performing certain common CMake tasks that don't #
# really fit into the other Macro files.                                       #
#                                                                              #
################################################################################

#
# Given a variable that contains a list of files (paths relative to the
# directory from which this macro is invoked), this macro will inplace (i.e.
# overwrite the input variable) convert the relative paths to full paths.
#
# Syntax: make_full_paths(<list>)
#     - list : A variable containing the list of files you want the full paths
#              for. This is NOT the files themselves, i.e. whatever you provide
#              must dereference to the list
#
function(make_full_paths __list)
    set(__prefix "${CMAKE_CURRENT_LIST_DIR}")
    foreach(__file ${${__list}})
        list(APPEND __temp_list ${__prefix}/${__file})
    endforeach()
    set(${__list} ${__temp_list} PARENT_SCOPE)
endfunction()
