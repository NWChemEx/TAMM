#include <TestBuildLibInt/TestBuildLibInt.hpp>
#include <catch/catch.hpp>

TEST_CASE("TestBuildLibInt")
{
    TestBuildLibInt test;
    REQUIRE(test.passed());
}
