################################################################################
#
#  These are macros to aid in debugging CMake scripts
#
################################################################################

#
# Prints the specified message iff NWX_DEBUG_CMAKE==TRUE
#
# Syntax: debug_message(<message_to_print>)
#     message_to_print : The actual message to print (i.e. not saved in a
#                        variable).
function(debug_message __msg)
    if(NWX_DEBUG_CMAKE)#Set in options section
        message(STATUS "${__msg}")
    endif()
endfunction()

#
# prints the path to the current file
#
# Syntax: debug_file_info()
#
function(debug_file_info)
    debug_message("In ${CMAKE_CURRENT_LIST_FILE}")
endfunction()

#
# prints the name of a variable and its current value
#
# Syntax: debug_print(<variable>)
function(debug_print _variable)
    debug_file_info()
    debug_message("\$\{${_variable}\} =(${${_variable}})")
endfunction()

