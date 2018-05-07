
The prerequisites needed to build TAMM can be found [here](dox/prerequisites.md) 

- TAMM uses [NWChemEx Base Repository](https://github.com/NWChemEx-Project/CMakeBuild) to manage the build process and can be built as follows:

```
TAMM_INSTALL_PATH=/opt/NWChemEx/install

git clone git@github.com:NWChemEx-Project/CMakeBuild.git
cd CMakeBuild
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran -DBUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH/CMakeBuild
make -j3
make install
```

```
git clone git@github.com:NWChemEx-Project/TAMM.git
cd TAMM
# Checkout the branch you want to build

mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran -DBUILD_TESTS=ON -DCMAKE_PREFIX_PATH=$TAMM_INSTALL_PATH/CMakeBuild -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH/TAMM -DBUILD_SHARED_LIBS=OFF 
make -j3
make install
```

