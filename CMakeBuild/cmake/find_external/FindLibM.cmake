# - Try to find the math library libm
#
#  In order to aid find_package the user may set LIBM_ROOT_DIR
#  to the root of the installed math library.
#
#   Once done this will define
#   LIBM_FOUND         - True if realtime library found.
#   LIBM_INCLUDE_DIRS  - where to find time.h, etc.
#   LIBM_LIBRARIES     - List of libraries when using libm.
#

find_path(LIBM_INCLUDE_DIRS
  NAMES math.h
  PATHS ${LIBM_ROOT_DIR}/include/
)

find_library(LIBM_LIBRARIES m)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibM DEFAULT_MSG LIBM_LIBRARIES LIBM_INCLUDE_DIRS)

mark_as_advanced(LIBM_INCLUDE_DIRS LIBM_LIBRARIES)