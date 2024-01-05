FROM ubuntu:latest
#
RUN mkdir -p /TAMM
WORKDIR /TAMM
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y git curl wget
RUN apt-get install -y python3-pip python3-dev python3-virtualenv
RUN pip3 install gcovr
RUN apt-get install -y gcc g++ gfortran
#RUN apt-get install -y libopenblas-base libopenblas-dev libgslcblas0 libgsl-dev liblapacke liblapacke-dev
#RUN apt-get install -y libboost-dev libboost-all-dev
RUN git clone https://github.com/NWChemEx-Project/TAMM.git
#
RUN apt-get install -y openmpi-bin libopenmpi-dev
#
WORKDIR /TAMM/TAMM
RUN git branch github-actions
RUN git checkout github-actions
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-linux-x86_64.sh
RUN yes | /bin/sh cmake-3.24.1-linux-x86_64.sh
RUN git clone https://github.com/NWChemEx-Project/CMakeBuild.git
RUN export INSTALL_PATH=`pwd`/install; cd CMakeBuild; ../cmake-3.24.1-Linux-x86_64/bin/cmake -H. -Bbuild -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}; cd build; make; make install
RUN export INSTALL_PATH=`pwd`/install; cmake-3.24.1-Linux-x86_64/bin/cmake -H. -Bbuild -DMPIEXEC_NUMPROC_FLAG="--allow-run-as-root -n" -DMPIEXEC_POSTFLAGS="--allow-run-as-root" -DCMAKE_PREFIX_PATH=${INSTALL_PATH}
# OpenMPI developers are allergic about anyone running programs as root.
# Now we have to set two environment variables to get around their sensitivities,
# even though in a Docker container this is the reasonable thing to do.
RUN export OMPI_ALLOW_RUN_AS_ROOT=1; export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; cd build; make VERBOSE=1; TAMM_CPU_POOL=20 ../cmake-3.24.1-Linux-x86_64/bin/ctest -VV
