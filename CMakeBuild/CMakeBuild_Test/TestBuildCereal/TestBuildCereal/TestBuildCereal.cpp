#include "TestBuildCereal/TestBuildCereal.hpp"
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>
#include <sstream>

bool TestBuildCereal::passed(){

    std::stringstream os;
    cereal::BinaryOutputArchive archive( os );
    double x{3.14};
    archive(x);

    return true;
}
