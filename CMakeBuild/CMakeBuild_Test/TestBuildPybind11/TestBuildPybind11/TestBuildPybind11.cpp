#include "TestBuildPybind11/TestBuildPybind11.hpp"
#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
m.doc() = "pybind11 example plugin"; // optional module docstring

m.def("add", &add, "A function which adds two numbers");
}

bool TestBuildPybind11::passed(){
    return true;
}
