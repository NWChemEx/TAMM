
The prerequisites needed to build TAMM can be found [here](dox/prerequisites.md) 

- TAMM uses [NWChemEx Base Repository](https://github.com/NWChemEx-Project/CMakeBuild) to manage the build process and can be built as follows:

```
TAMM_INSTALL_PATH=/opt/NWChemEx/install

git clone git@github.com:NWChemEx-Project/CMakeBuild.git
cd CMakeBuild
mkdir build && cd build
cmake .. -DBUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH/CMakeBuild
make -j3
make install
```

```
git clone git@github.com:NWChemEx-Project/TAMM.git
cd TAMM
git checkout tamm
mkdir build && cd build
#Add -DENABLE_GPU=ON below if you want to build TAMM with GPU support
cmake .. -DBUILD_TESTS=OFF -DCMAKE_PREFIX_PATH=$TAMM_INSTALL_PATH/CMakeBuild -DCMAKE_INSTALL_PREFIX=$TAMM_INSTALL_PATH/TAMM -DBUILD_SHARED_LIBS=OFF 
make -j3
make install
```

