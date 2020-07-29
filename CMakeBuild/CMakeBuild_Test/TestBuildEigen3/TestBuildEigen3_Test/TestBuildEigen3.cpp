#include <TestBuildEigen3/TestBuildEigen3.hpp>
#include <catch/catch.hpp>

TEST_CASE("TestBuildEigen3")
{
    TestBuildEigen3 test;
    REQUIRE(test.passed());
}
