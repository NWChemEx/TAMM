#define CATCH_CONFIG_MAIN
#include <catch/catch.hpp>

#include <tamm/labeled_tensor.hpp>
#include <iostream>
#include <string>

using namespace tamm;

template<typename T>
std::ostream&
operator << (std::ostream& os, const std::vector<T>& vec) {
  os<<"[";
  for(const auto& v: vec) {
    os<<v<<",";
  }
  os<<"]"<<std::endl;
  return os;
}

TEST_CASE("Zero-dimensional tensor with double") {
  bool failed = false;
  //IndexSpace is {range(10)};
  //TiledIndexSpace tis{is};
  Tensor<double> T1{};
  try {
    T1() = double{0};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
}

TEST_CASE("Zero-dimensional tensor with int") {
  bool failed = false;
  //IndexSpace is {range(10)};
  //TiledIndexSpace tis{is};
  Tensor<double> T1{};
  try {
    T1() = int{0};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
}

/** @todo This is a compile-time failing test. Need to enable it*/
// TEST_CASE("Zero-dimensional tensor with string") {
//   bool failed = false;
//   //IndexSpace is {range(10)};
//   //TiledIndexSpace tis{is};
//   Tensor<double> T1{};
//   try {
//     T1() = std::string{"fails"};
//   } catch (...) {
//     failed = true;
//   }
//   REQUIRE(failed);
// }

TEST_CASE("One-dimensional tensor with double") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis};
  try {
    T1() = double{7};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;

  try {
    T1("i") = double{0};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;

  TiledIndexLabel i = tis.label("all");
  try {
    T1(i) = double{5};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
}

TEST_CASE("One-dimensional tensor with int") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis};
  try {
    T1() = int{7};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;

  try {
    T1("i") = int{0};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;

  TiledIndexLabel i = tis.label("all");
  try {
    T1(i) = int{5};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
}

TEST_CASE("Two-dimensional tensor") {
  bool failed = false;
  IndexSpace is {range(10)};
  TiledIndexSpace tis{is};
  Tensor<double> T1{tis, tis};
  TiledIndexLabel i, j;
  std::tie(i, j) = tis.labels<2>("all");

  try {
    T1() = double{8};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;

  try {
    T1(i, j) = double{5};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;

  try {
    T1(i, "j") = double{5};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;

  try {
    T1(i, "i") = double{5};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;

  try {
    T1("x", "x") = double{5};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;

  try {
    T1(i, i) = double{5};
  } catch (...) {
    failed = true;
  }
  REQUIRE(!failed);
  failed = false;

  //invalid operations: should fail

  //invalid number of labels
  try {
    T1(i) = double{8};
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;

  //invalid number of labels
  try {
    T1("x") = double{8};
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;

  //invalid labels
  try {
    T1(i, i(j)) = double{8};
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;

  //invalid labels
  try {
    T1(i, j(i)) = double{8};
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;

  //invalid number of labels
  try {
    T1(i, j, "x") = double{8};
  } catch (...) {
    failed = true;
  }
  REQUIRE(failed);
  failed = false;
}