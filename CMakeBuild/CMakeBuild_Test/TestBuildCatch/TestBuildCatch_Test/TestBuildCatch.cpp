#include <TestBuildCatch/TestBuildCatch.hpp>
#include <catch/catch.hpp>

TEST_CASE("TestBuildCatch")
{
    TestBuildCatch test;
    REQUIRE(test.passed());
}
