#include <TestBuildLAPACKE/TestBuildLAPACKE.hpp>
#include <catch/catch.hpp>

TEST_CASE("TestBuildLAPACKE")
{
    TestBuildLAPACKE test;
    REQUIRE(test.passed());
}
