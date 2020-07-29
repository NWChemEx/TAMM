#include <TestBuildBPHash/TestBuildBPHash.hpp>
#include <catch/catch.hpp>

TEST_CASE("TestBuildBPHash")
{
    TestBuildBPHash test;
    REQUIRE(test.passed());
}
