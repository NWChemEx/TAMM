cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(NWTEST VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For nwchemex_add_executable

set(NWTEST_SRCS 
put_hash_block.F
add_block.F
get_block.F
get_block_ind.F
add_hash_block.F
get_hash_block.F
get_hash_block_ma.F
nxtask.F
nxtval_ga.F
#errquit.F
put_block.F
tce_hash.F
tce_hash_ind.F
tce_restricted.F
tce_sort.F
tce_sortacc.F
misc.F
ccsd_t1.F
ccsd_t2.F
cc2_t1.F
cc2_t2.F
set_vars.F
)

set(NWTEST_INCLUDES )
set(NWTEST_DEFINITONS )
set(NWTEST_LFLAGS )


nwchemex_add_library(nwtest NWTEST_SRCS NWTEST_INCLUDES NWTEST_DEFINITONS
                                    NWTEST_LFLAGS)
