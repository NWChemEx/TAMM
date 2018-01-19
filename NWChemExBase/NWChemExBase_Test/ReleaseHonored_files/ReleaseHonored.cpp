#include "ReleaseHonored.hpp"

void ReleaseHonored::run_test()
{
#ifndef NDEBUG
    throw std::runtime_error("Errr... in Debug mode");
#endif
}
