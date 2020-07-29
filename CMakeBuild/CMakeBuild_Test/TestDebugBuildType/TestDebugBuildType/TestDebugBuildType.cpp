#include "TestDebugBuildType/TestDebugBuildType.hpp"
#include <stdexcept>

bool TestDebugBuildType::passed(){

#ifdef NDEBUG
        throw std::runtime_error("Errr... not in Debug mode");
#endif
    return true;
}
