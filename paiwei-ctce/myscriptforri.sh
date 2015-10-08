export NWCHEM_TOP=$PWD
export NWCHEM_MODULES="all"
export NWCHEM_TARGET=LINUX64
export USE_SUBGROUPS=yes
export LARGE_FILES=TRUE

export FOPTIONS="-O3 -i8"
export FOPT="-O3 -i8"
export COPTIONS="-O3"
export COPT="-O3"
#export FC="ifort"
#export CC="icc"

export USE_MPI=y
export MPI_LOC="/home/arafatm/common/mvapich-1.9"
export MPI_LIB=$MPI_LOC/lib
export MPI_INCLUDE=$MPI_LOC/include
export LIBMPI="-lmpich -lirc"

# === Infiniband ===
#export IB_HOME="/usr"
#export IB_INCLUDE=$IB_HOME/include/infiniband
#export IB_LIB=$IB_HOME/lib64
#export IB_LIB_NAME="-lrdmacm -libverbs -libumad -lpthread -lpthread -lrt"
export ARMCI_NETWORK=OPENIB

# === GA Tools ===
export OLD_GA=y
export TARGET=LINUX64
export TARGET_CPU=x86_64

# === SLURM ===
export SLURM=y
export SLURMOPT="-L/usr/lib64 -lpmi"

#make nwchem_config 
#make FC=ifort CC=icc DNTMC_TOR=y 2>&1|tee makelog


