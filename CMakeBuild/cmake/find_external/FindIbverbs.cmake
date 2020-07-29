# - Try to find ibverbs libraries
#
#  In order to aid find_package the user may set IBVERBS_ROOT_DIR to the root of
#  the installed library.
#
#   Once done this will define
#   IBVERBS_FOUND         - True if ibverbs library found.
#   IBVERBS_INCLUDE_DIRS  - where to find verbs.h
#   IBVERBS_LIBRARIES     - List of libraries 
#

find_package(PkgConfig QUIET)
pkg_check_modules(PC_IBVERBS QUIET libibverbs)

find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h
  HINTS
  ${IBVERBS_ROOT_DIR} ENV IBVERBS_ROOT_DIR
  ${PC_IBVERBS_INCLUDEDIR}
  ${PC_IBVERBS_INCLUDE_DIRS}
  PATH_SUFFIXES include)

find_library(IBVERBS_LIBRARY NAMES ibverbs libibverbs
  HINTS
    ${IBVERBS_ROOT_DIR} ENV IBVERBS_ROOT_DIR
    ${PC_IBVERBS_LIBDIR}
    ${PC_IBVERBS_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

set(IBVERBS_LIBRARIES ${IBVERBS_LIBRARY} CACHE INTERNAL "")
set(IBVERBS_INCLUDE_DIRS ${IBVERBS_INCLUDE_DIR} CACHE INTERNAL "")

find_package_handle_standard_args(Ibverbs DEFAULT_MSG
  IBVERBS_LIBRARY IBVERBS_INCLUDE_DIR)

mark_as_advanced(IBVERBS_ROOT_DIR IBVERBS_LIBRARY IBVERBS_INCLUDE_DIR)
