include(AssertMacros.cmake)

#There's no way to signal that the commented lines are intentional failures
#uncomment o manually check

set(this_is_true TRUE)
assert(this_is_true)
#assert(NOT this_is_true)

set(this_is_false FALSE)
assert(NOT this_is_false)
#assert(this_is_false)

assert("LHS" STREQUAL "LHS")
#assert("LHS" STREQUAL "RHS")

assert_strings_are_equal("LHS" "LHS")
#assert_strings_are_equal("LHS" "RHS")
