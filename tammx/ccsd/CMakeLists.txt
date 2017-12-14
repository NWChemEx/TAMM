include(UtilityMacros) # For prefix_paths

set(TAMM_SRCS ccsd_driver.cc)
prefix_paths(ccsd TAMM_SRCS)
set(TAMM_SRCS ${TAMM_SRCS} PARENT_SCOPE)
