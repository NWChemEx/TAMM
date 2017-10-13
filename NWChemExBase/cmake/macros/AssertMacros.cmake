################################################################################
#                                                                              #
# These are macros for asserting conditions are met.  They largely serve as    #
# wrappers for pretty printing.                                                #
#                                                                              #
################################################################################

#
# Asserts that the result of comparing two variables is true.
#
# Syntax: assert(<LHS> <Comp> <RHS>)
#    - LHS  : The variable on the left side of the operator
#    - Comp : The operator to use for the comparison
#    - RHS  : The variable on the right side of the operator
#
function(assert __lhs __comp __rhs)
    if(NOT ${${__lhs}} ${__comp} ${${__rhs}})
        message(FATAL_ERROR "Assertion: ${__lhs} ${__comp} ${__rhs} failed.\n"
                            "    ${__lhs}:${${__lhs}}\n"
                            "    ${__rhs}:${${__rhs}}")
    endif()
endfunction()

function(assert_strings_are_equal __lhs __rhs)
    assert(${__lhs} STREQUAL ${__rhs})
endfunction()
