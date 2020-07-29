# - Try to find POSIX.1b Realtime Extensions library
#
#  In order to aid find_package the user may set LIBRT_ROOT_DIR to the root of
#  the installed realtime library.
#
#   Once done this will define
#   LIBRT_FOUND         - True if realtime library found.
#   LIBRT_INCLUDE_DIRS  - where to find time.h, etc.
#   LIBRT_LIBRARIES     - List of libraries when using librt.
#

find_path(LIBRT_INCLUDE_DIRS
  NAMES time.h
  PATHS ${LIBRT_ROOT_DIR}/include/
)

find_library(LIBRT_LIBRARIES rt)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibRT DEFAULT_MSG LIBRT_LIBRARIES LIBRT_INCLUDE_DIRS)

mark_as_advanced(LIBRT_INCLUDE_DIRS LIBRT_LIBRARIES)