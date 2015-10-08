export NWCHEM_TOP=/home/laip/nwchem-6.3
export NWCHEM_TARGET=LINUX64

cd $NWCHEM_TOP/src
make CC=mpicc FC=gfortran nwchem_config 
make
