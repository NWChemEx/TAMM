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
    touch $1/src/tce/ccsd_energy_loc.F
    touch $1/src/tce/ccsd_lambda.F
    touch $1/src/tce/eaccsd/eaccsd_x1.F
    touch $1/src/tce/eaccsd/eaccsd_x2.F
    touch $1/src/tce/eaccsd/tce_eax1_offset.F
    touch $1/src/tce/eaccsd/tce_eax2_offset.F
    touch $1/src/tce/eaccsd/tce_eom_eaxguess.F
    touch $1/src/tce/eaccsd/tce_jacobi_eax1.F
    touch $1/src/tce/eaccsd/tce_jacobi_eax2.F
    touch $1/src/tce/eaccsd/tce_print_eax1.F
    touch $1/src/tce/eaccsd/tce_print_eax2.F
    touch $1/src/tce/ipccsd/ipccsd_x1.F
    touch $1/src/tce/ipccsd/ipccsd_x2.F
    touch $1/src/tce/ipccsd/tce_eom_ipxguess.F
    touch $1/src/tce/ipccsd/tce_ipx1_offset.F
    touch $1/src/tce/ipccsd/tce_ipx2_offset.F
    touch $1/src/tce/ipccsd/tce_jacobi_ipx1.F
    touch $1/src/tce/ipccsd/tce_jacobi_ipx2.F
    touch $1/src/tce/ipccsd/tce_print_ipx1.F
    touch $1/src/tce/ipccsd/tce_print_ipx2.F
    touch $1/src/tce/tce_energy.F 
fi
#cp *.F $1/src/tce/

