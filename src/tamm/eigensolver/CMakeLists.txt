cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(es VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For nwchemex_add_library

include_directories(include)

#Add the current directory's sources to the list
set(es_SRCS 
    src/cholQR.cpp 
    src/sisubit.cpp 
    src/RayleighRitz.cpp 
    src/GetEigvals.cpp 
    src/GetResiduals.cpp 
    src/itereig.cpp 
    src/evalidate.cpp 
    src/HSLanczos.cpp 
    src/getevbnd.cpp 
    src/parseinput.cpp 
    src/utilities.cpp 
    src/getshifts.cpp 
    src/slicing.cpp 
    src/collectevs.cpp 
    src/sync_slices.cpp 
    src/hsdiag.cpp 
    src/kmeans.cpp
)

#Add the current directory's header files to the list
set(es_include
    include/collectevs.hpp
    include/enforcecounts.hpp
    include/evalidate.hpp
    include/getshifts.hpp
    include/itereig.hpp
    include/parseinput.hpp
    include/partev.hpp
    include/sisubit.hpp
    include/slicing.hpp
    include/sync_slices.hpp
    include/utilities.hpp
)

set(es_DEFINITIONS )
set(es_LFLAGS )

nwchemex_add_library(es es_SRCS es_include es_DEFINTIONS es_LFLAGS)


