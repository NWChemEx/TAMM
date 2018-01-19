#include "SkipTests/SkipTests.hpp"
#include<stdexcept>

void SkipTests::run_test()
{
    throw std::runtime_error("Guess the tests weren't skipped...");
}

