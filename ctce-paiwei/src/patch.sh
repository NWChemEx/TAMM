# backup original NWChem-6.0 files
cp ../src/tce/ccsd_t/ccsd_t.F ../src/tce/ccsd_t/ccsd_t.F.old
cp ../src/tce/ccsd/ccsd_t1.F ../src/tce/ccsd/ccsd_t1.F.old
cp ../src/tce/ccsd/ccsd_t2.F ../src/tce/ccsd/ccsd_t2.F.old

# patch cTCE files
cp patchfile/GNUmakefile_for_src ../src/GNUmakefile
cp patchfile/ccsd_t.F ../src/tce/ccsd_t/ccsd_t.F
cp patchfile/ccsd_t1.F ../src/tce/ccsd/ccsd_t1.F
cp patchfile/ccsd_t2.F ../src/tce/ccsd/ccsd_t2.F

