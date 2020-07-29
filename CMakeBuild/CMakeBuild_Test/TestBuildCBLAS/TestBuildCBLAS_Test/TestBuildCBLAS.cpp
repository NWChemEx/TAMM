#include <TestBuildCBLAS/TestBuildCBLAS.hpp>
#include <catch/catch.hpp>

TEST_CASE("TestBuildCBLAS")
{
    TestBuildCBLAS test;
    REQUIRE(test.passed());
}
