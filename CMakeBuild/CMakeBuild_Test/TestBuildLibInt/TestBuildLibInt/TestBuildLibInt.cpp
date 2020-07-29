#include "TestBuildLibInt/TestBuildLibInt.hpp"
#include <libint2.hpp>

bool TestBuildLibInt::passed(){
    libint2::initialize();
    libint2::finalize();
    return true;
}
