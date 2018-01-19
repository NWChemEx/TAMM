#include "DebugHonored.hpp"

void DebugHonored::run_test()
{
#ifdef NDEBUG
    throw std::runtime_error("Errr... not in Debug mode");
#endif
}
