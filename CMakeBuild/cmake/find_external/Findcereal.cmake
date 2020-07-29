#
# Cereal's config file only makes a target and not the includes
#

find_package(cereal CONFIG)
is_valid_and_true(cereal_FOUND found_cereal)
if(found_cereal)
    get_property(c_ing TARGET cereal PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    set(CEREAL_INCLUDE_DIRS ${c_ing})
endif()
