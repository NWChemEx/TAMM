
export NWCHEM_TOP=/home/laip/nwchem-6.3
export NWCHEM_TARGET=LINUX64

cd $NWCHEM_TOP/bin/LINUX64
rm nwchem

cd $NWCHEM_TOP/ctce
make clean
make

cd $NWCHEM_TOP/src/tce/ccsd_t
touch ccsd_t.F
make

cd $NWCHEM_TOP/src/tce/ccsd
touch ccsd_t1.F
touch ccsd_t2.F
make

cd $NWCHEM_TOP/src
make nwchem_config NWCHEM_MODULES=qm
make link

