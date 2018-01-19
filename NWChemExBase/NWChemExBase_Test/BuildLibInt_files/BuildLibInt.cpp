#include "BuildLibInt/BuildLibInt.hpp"
#include <libint2.hpp>

void BuildLibInt::run_test()
{
    libint2::initialize();
    libint2::finalize();
}
