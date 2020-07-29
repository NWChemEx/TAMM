#include <TestBuildCereal/TestBuildCereal.hpp>
#include <catch/catch.hpp>

TEST_CASE("TestBuildCereal")
{
    TestBuildCereal test;
    REQUIRE(test.passed());
}
