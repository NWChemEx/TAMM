#include <TestSharedLibrary/TestSharedLibrary.hpp>
#include <catch/catch.hpp>

TEST_CASE("TestSharedLibrary")
{
    TestSharedLibrary test;
    REQUIRE(test.value_ == 2);
}
