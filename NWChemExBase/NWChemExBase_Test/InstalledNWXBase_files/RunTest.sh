#!/usr/bin/env bash
binary_dir=$(pwd)
cd ../..
nwx_path=$(pwd)
cd -
cd InstalledNWXBase_files
dummy_repo=$(pwd)/DummyRepo_files

#Build and install dummy repo
${binary_dir}/RunNWChemExBaseTest.sh DummyRepo ${nwx_path} ${dummy_repo} ${1}
cd DummyRepo_repo_test/build
VERBOSE=1 make install

#Try building the test
my_files=${binary_dir}/InstalledNWXBase_files
cd ${binary_dir}
./RunNWChemExBaseTest.sh InstalledNWXBase ${nwx_path} ${my_files} ${1}
