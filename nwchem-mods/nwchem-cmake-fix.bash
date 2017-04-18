#!/bin/bash
if [ $# -gt 0 ]; then
    sed -i -- "s:(TAMM_LIB_PATH):$2:g" $1/CMakeLists.txt
    sed -i -- "s:(TAMM_LIBS):$3:g" $1/CMakeLists.txt
    sed -i -- "s:#set(TAMM_LIBRARIES:set(TAMM_LIBRARIES:g" $1/CMakeLists.txt
else
    echo "Usage $0 INSTALL_LIB_PATH"
fi

if [ -z "$4" ]; then
    rsync -rav --progress ./src $1 
fi
