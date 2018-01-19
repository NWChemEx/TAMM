# Find NWX_Catch
#
# Catch doesn't by default install itself.  This script will find the
# Catch header file (catch/catch.hpp) and the catch library (catch/libcatch.so).
# The NWX just reminds us this isn't the canonical way provided by Catch
#
# This module defines
#  NWX_CATCH_INCLUDE_DIRS, where to find catch/catch.hpp
#  NWX_CATCH_LIBRARIES, where to find libcatch.so
#  NWX_Catch_FOUND, True if we found NWX_Catch

find_path(NWX_CATCH_INCLUDE_DIR catch/catch.hpp)
find_library(NWX_CATCH_LIBRARY libcatch${CMAKE_SHARED_LIBRARY_SUFFIX})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NWX_Catch DEFAULT_MSG NWX_CATCH_INCLUDE_DIR
                                                        NWX_CATCH_LIBRARY)
set(NWX_CATCH_FOUND ${NWX_Catch_FOUND})
set(NWX_CATCH_LIBRARIES ${NWX_CATCH_LIBRARY})
set(NWX_CATCH_INCLUDE_DIRS ${NWX_CATCH_INCLUDE_DIR})


