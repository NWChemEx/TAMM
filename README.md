
The prerequisites needed to build TAMM can be found [here](old_files/README.md) 

Currently, TAMM has two CMake builds:

- The old CMake build described [here](old_files/README.md)

- The latest CMake build using [NWChemEx base repository](https://github.com/NWChemEx-Project/CMakeBuild) can be used as follows:

```
git clone git@github.com:NWChemEx-Project/CMakeBuild.git
cd CMakeBuild
mkdir build && cd build
cmake .. -DBUILD_TESTS=OFF  -DCMAKE_INSTALL_PREFIX=/opt/NWChemEx/install/CMakeBuild
make -j3
make install
```

```
git clone git@github.com:NWChemEx-Project/TAMM.git
cd TAMM
git checkout devel
mkdir build && cd build
cmake .. -DBUILD_TESTS=OFF -DCMAKE_PREFIX_PATH=/opt/NWChemEx/install/CMakeBuild -DCMAKE_INSTALL_PREFIX=/opt/NWChemEx/install/TAMM
make -j3
make install
```

