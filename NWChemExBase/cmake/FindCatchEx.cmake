# Find CatchEx
#
# Catch doesn't by default install itself.  This script will find the
# Catch header file (catch/catch.hpp) and the catch library (catch/libcatch.so)
#
# This module defines
#  Catch_INCLUDE_DIRS, where to find catch/catch.hpp
#  Catch_LIBARIES, where to find libcatch.so
#  Catch_FOUND, True if we found Catch

find_path(CatchEx_INCLUDE_DIR catch/catch.hpp)
#find_path(Catch_LIBRARY libcatch.so)

find_package_handle_standard_args(CatchEx DEFAULT_MSG CatchEx_INCLUDE_DIR)
#                                                    Catch_LIBRARY)

set(CatchEx_INCLUDE_DIRS ${CatchEx_INCLUDE_DIR})
#set(Catch_LIBRARIES Catch_LIBRARY)
