#include <TestBuildPybind11/TestBuildPybind11.hpp>
#include <catch/catch.hpp>

TEST_CASE("TestBuildPybind11")
{
    TestBuildPybind11 test;
    REQUIRE(test.passed());
}
