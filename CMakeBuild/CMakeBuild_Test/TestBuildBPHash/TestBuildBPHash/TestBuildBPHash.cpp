#include "TestBuildBPHash/TestBuildBPHash.hpp"
#include <bphash/Hasher.hpp>
using namespace bphash;

bool TestBuildBPHash::passed(){
    int i = 1921;;
    double f = 1.234;
    HashValue hv1 = make_hash(HashType::Hash128, i);
    HashValue hv2 = make_hash(HashType::Hash128, f);
    HashValue hv3 = make_hash(HashType::Hash128, i, f);
    return true;
}
