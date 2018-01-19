################################################################################
#                                                                              #
# These are macros for asserting conditions are met.  They largely serve as    #
# wrappers for pretty printing.                                                #
#                                                                              #
################################################################################

#
# Asserts that the condition passed to this function is true.
#
# Syntax:  assert(<condition_to_check>)
#
function(assert)
    if(${ARGN})
        #Empty to avoid two NOTs being next to each other...
    else()
        message(FATAL_ERROR "Assertion: ${ARGN} failed.\n")
    endif()
endfunction()

function(assert_strings_are_equal __lhs __rhs)
    assert("${__lhs}" STREQUAL "${__rhs}")
endfunction()
