#include <TestReleaseBuildType/TestReleaseBuildType.hpp>
#include <catch/catch.hpp>

TEST_CASE("TestReleaseBuildType")
{
    TestReleaseBuildType test;
    REQUIRE(test.passed());
}
