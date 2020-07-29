#include "TestReleaseBuildType/TestReleaseBuildType.hpp"
#include <stdexcept>

bool TestReleaseBuildType::passed(){
#ifndef NDEBUG
    throw std::runtime_error("Errr... in Debug mode");
#endif
    return true;
}
