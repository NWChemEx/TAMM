# Find BerkeleyDB
# Find the BerkeleyDB includes and library
#
# This module defines
#  BERKELEYDB_INCLUDE_DIRS, where to find db.h, etc.
#  BERKELEYDB_LIBRARIES, the libraries needed to use BerkeleyDB.
#  BERKELEYDB_FOUND, If false, do not try to use BerkeleyDB.

find_path(BERKELEYDB_INCLUDE_DIR db.h)

find_library(BERKELEYDB_LIBRARY NAMES db)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BERKELEYDB DEFAULT_MSG BERKELEYDB_INCLUDE_DIR
                                                         BERKELEYDB_LIBRARY
)

mark_as_advanced(BERKELEYDB_LIBRARY BERKELEYDB_INCLUDE_DIR)
set(BERKELEYDB_LIBRARIES ${BERKELEYDB_LIBRARY})
set(BERKELEYDB_INCLUDE_DIRS ${BERKELEYDB_INCLUDE_DIR})

add_library(BerkeleyDB INTERFACE)
target_link_libraries(BerkeleyDB INTERFACE ${BERKELEYDB_LIBRARIES})
target_include_directories(BerkeleyDB INTERFACE ${BERKELEYDB_INCLUDE_DIRS})
