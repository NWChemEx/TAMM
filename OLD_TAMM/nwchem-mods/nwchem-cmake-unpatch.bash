#!/bin/bash
cd $1/
git checkout CMakeLists.txt
svn revert src/tce/ccsd_energy_loc.F
svn revert src/tce/ccsd_lambda.F
svn revert src/tce/eaccsd/eaccsd_x1.F
svn revert src/tce/eaccsd/eaccsd_x2.F
svn revert src/tce/eaccsd/tce_eax1_offset.F
svn revert src/tce/eaccsd/tce_eax2_offset.F
svn revert src/tce/eaccsd/tce_eom_eaxguess.F
svn revert src/tce/eaccsd/tce_jacobi_eax1.F
svn revert src/tce/eaccsd/tce_jacobi_eax2.F
svn revert src/tce/eaccsd/tce_print_eax1.F
svn revert src/tce/eaccsd/tce_print_eax2.F
svn revert src/tce/ipccsd/ipccsd_x1.F
svn revert src/tce/ipccsd/ipccsd_x2.F
svn revert src/tce/ipccsd/tce_eom_ipxguess.F
svn revert src/tce/ipccsd/tce_ipx1_offset.F
svn revert src/tce/ipccsd/tce_ipx2_offset.F
svn revert src/tce/ipccsd/tce_jacobi_ipx1.F
svn revert src/tce/ipccsd/tce_jacobi_ipx2.F
svn revert src/tce/ipccsd/tce_print_ipx1.F
svn revert src/tce/ipccsd/tce_print_ipx2.F
svn revert src/tce/tce_energy.F
cd -

