FROM ubuntu:latest
ARG github_token
#
# Github_token is either a Github Access token or simply
# <username>:<password>. Either way simply substituting
# git clone https://<github_token>@github.com/<owner>/<repo>
# should work.
#
RUN mkdir -p /TAMM
WORKDIR /TAMM
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y git curl wget
RUN apt-get install -y python3-pip python3-dev python3-virtualenv
RUN pip3 install gcovr
RUN apt-get install -y gcc-8 g++-8 gfortran-8
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 95 --slave /usr/bin/g++ g++ /usr/bin/g++-8 --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-8 --slave /usr/bin/gcov gcov /usr/bin/gcov-8
RUN update-alternatives --install /usr/bin/cpp cpp /usr/bin/cpp-8 95
RUN apt-get install -y libopenblas-base libopenblas-dev libgslcblas0 libgsl-dev liblapacke liblapacke-dev
RUN apt-get install -y libboost-dev libboost-all-dev
RUN apt-get install -y libeigen3-dev
RUN git clone https://${github_token}@github.com/hjjvandam/TAMM.git
#
# The regular OpenMPI does not allow you to run MPI programs as root.
# So you cannot use that in a Docker container, and we have to install
# the latest OpenMPI 4 from source. (02/12/2020). 
# However, installing OpenMPI from source somehow does not work, 
# as it leads to failing MPI_Init problems. I will have to hack the
# mpiexec invocations instead.
#
RUN apt-get install -y openmpi-bin libopenmpi-dev
#WORKDIR /
#RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.2.tar.gz
#RUN tar -zxf openmpi-4.0.2.tar.gz
#RUN cd openmpi-4.0.2; ./configure --prefix=/usr; make; make install
#
WORKDIR /TAMM/TAMM
RUN git branch github-actions
RUN git checkout github-actions
RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.3/cmake-3.16.3-Linux-x86_64.sh
RUN yes | /bin/sh cmake-3.16.3-Linux-x86_64.sh
RUN git clone https://${github_token}@github.com/NWChemEx-Project/CMakeBuild.git
RUN export INSTALL_PATH=`pwd`/install; cd CMakeBuild; ../cmake-3.16.3-Linux-x86_64/bin/cmake -H. -Bbuild -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH}; cd build; make; make install
RUN export INSTALL_PATH=`pwd`/install; cmake-3.16.3-Linux-x86_64/bin/cmake -H. -Bbuild -DBUILD_TESTS=ON -DCATCH_ENABLE_COVERAGE=ON -DMPIEXEC_POSTFLAGS="--allow-run-as-root" -DCMAKE_CXX_FLAGS="-O0 --coverage" -DCMAKE_C_FLAGS="-O0 --coverage" -DCMAKE_Fortran_FLAGS="-O0 --coverage" -DCMAKE_EXE_LINKER_FLAGS="-O0 -fprofile-arcs" -DCMAKE_PREFIX_PATH=${INSTALL_PATH}
# OpenMPI developers are allergic about anyone running programs as root.
# Now we have to set two environment variables to get around their sensitivities,
# even though in a Docker container this is the reasonable thing to do.
RUN export OMPI_ALLOW_RUN_AS_ROOT=1; export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; cd build; make; ../cmake-3.16.3-Linux-x86_64/bin/ctest -VV
