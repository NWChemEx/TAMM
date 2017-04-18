#!/bin/bash
if [ $# -gt 0 ]; then
    cp template.patch gnumake.patch
    sed -i -- "s:(LIB_TEMP):$2:g" ./gnumake.patch
    sed -i -- "s:(LIB_ANTLR):$3:g" ./gnumake.patch
    patch -s -N -r - $1/src/GNUmakefile ./gnumake.patch
    rm gnumake.patch
else
    echo "Usage $0 INSTALL_LIB_PATH"
fi

if [ -z "$4" ]; then
    rsync -rav --progress ./src $1
fi
#cp *.F $1/src/tce/

