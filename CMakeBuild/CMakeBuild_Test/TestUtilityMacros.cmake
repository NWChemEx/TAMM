include(UtilityMacros)
include(AssertMacros)

#is_valid
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

#is_valid_and_true
set(is_valid_false FALSE)
is_valid_and_true(is_valid_false test5)
assert(NOT test5)
set(is_valid_true TRUE)
is_valid_and_true(is_valid_true test6)
assert(test6)

#prefix_paths
set(A_LIST file1.h file2.cc)
prefix_paths(/home A_LIST)
list(GET A_LIST 0 test7)
assert_strings_are_equal(${test7} "/home/file1.h")
list(GET A_LIST 1 test8)
assert_strings_are_equal(${test8} "/home/file2.cc")
set(prefix_ /home)
prefix_paths(${prefix_} A_LIST)
list(GET A_LIST 0 test9)
assert_strings_are_equal(${test9} "/home//home/file1.h")
list(GET A_LIST 1 test10)
assert_strings_are_equal(${test10} "/home//home/file2.cc")

#make_full_paths
set(A_LIST file1.h file2.cc)
make_full_paths(A_LIST)
list(GET A_LIST 0 test11)
assert_strings_are_equal(${test11} "${CMAKE_CURRENT_LIST_DIR}/file1.h")
list(GET A_LIST 1 test12)
assert_strings_are_equal(${test12} "${CMAKE_CURRENT_LIST_DIR}/file2.cc")

#clean_flags
set(SOME_FLAGS "-O3")
list(APPEND SOME_FLAGS "-fPIC" "-O3")
clean_flags(SOME_FLAGS test13)
assert_strings_are_equal(${test13} "-O3 -fPIC")


#string_concat
set(SOME_LIST "thing1" "thing2")
string_concat(SOME_LIST "" "" test14)
assert_strings_are_equal(${test14} "thing1thing2")
string_concat(SOME_LIST "-I" "" test15)
assert_strings_are_equal(${test15} "-Ithing1-Ithing2")
string_concat(SOME_LIST "-I" "+" test16)
assert_strings_are_equal(${test16} "-Ithing1+-Ithing2")
