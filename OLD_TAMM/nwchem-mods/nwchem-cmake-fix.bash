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
