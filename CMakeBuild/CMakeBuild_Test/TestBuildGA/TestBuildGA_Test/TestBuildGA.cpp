#include <TestBuildGA/TestBuildGA.hpp>
#include <catch/catch.hpp>

TEST_CASE("TestBuildGA")
{
    TestBuildGA test;
    REQUIRE(test.passed());
}
