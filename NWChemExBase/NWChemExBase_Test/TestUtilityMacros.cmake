include(UtilityMacros.cmake)
include(AssertMacros.cmake)

is_valid(not_defined test1)
assert(NOT test1)

set(blank_string "")
is_valid(blank_string test2)
assert(NOT test2)

set(valid_string "this_is_valid")
is_valid(valid_string test3)
assert(test3)

set(valid_list item1 item2)
is_valid(valid_list test4)
assert(test4)

set(is_valid_false FALSE)
is_valid_and_true(is_valid_false test5)
assert(NOT test5)

set(is_valid_true TRUE)
is_valid_and_true(is_valid_true test6)
assert(test6)

set(A_LIST file1.h file2.cc)
prefix_paths(/home A_LIST)
list(GET A_LIST 0 value)
assert_strings_are_equal(${value} "/home/file1.h")
list(GET A_LIST 1 value)
assert_strings_are_equal(${value} "/home/file2.cc")
set(prefix_ /home)
prefix_paths(${prefix_} A_LIST)
list(GET A_LIST 0 value)
assert_strings_are_equal(${value} "/home//home/file1.h")
list(GET A_LIST 1 value)
assert_strings_are_equal(${value} "/home//home/file2.cc")

set(SOME_FLAGS "-O3")
list(APPEND SOME_FLAGS "-fPIC" "-O3")
clean_flags(SOME_FLAGS CLEAN_FLAGS)
assert_strings_are_equal(${CLEAN_FLAGS} "-O3 -fPIC")
