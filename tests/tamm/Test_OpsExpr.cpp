#define CATCH_CONFIG_MAIN

#include "catch/catch.hpp"
#include "ga.h"
#include "mpi.h"
#include "macdecls.h"
#include "ga-mpi.h"
#include "tamm/tamm.hpp"

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

TEST_CASE("SCF commutator declarations") {
    bool failed = false;
    try {
        using tensor_type = tamm::Tensor<double>;
        using space_type = tamm::TiledIndexSpace;
        using index_type = tamm::TiledIndexLabel;

        tensor_type comm, temp, F, D, S;
        space_type AOs = D.get_spaces()[0];
        index_type mu, nu, lambda;

        temp(mu, lambda) = F(mu,nu)*D(nu, lambda); //FD
        comm(mu, lambda) = temp(mu, nu)*S(nu, lambda); //FDS
        temp(mu, lambda) = S(mu, nu)*D(nu, lambda); //SD
        comm(mu, lambda) += -1.0*temp(mu, nu)*F(nu, lambda);//FDS - SDF

    } catch (...) {
        failed = true;
    }
    REQUIRE(failed);
}
