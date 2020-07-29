#include <TestDebugBuildType/TestDebugBuildType.hpp>
#include <catch/catch.hpp>

TEST_CASE("TestDebugBuildType")
{
    TestDebugBuildType test;
    REQUIRE(test.passed());
}
