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

TEST_CASE("SCF Commutator declarations") {
    bool failed = false;
    try {
        using tensor_type = tamm::Tensor<double>;
        using space_type = tamm::TiledIndexSpace;
        using index_type = tamm::TiledIndexLabel;

        tensor_type comm, temp, F, D, S;
        index_type mu, nu, lambda;

        temp(mu, lambda) = F(mu,nu)*D(nu, lambda); //FD
        comm(mu, lambda) = temp(mu, nu)*S(nu, lambda); //FDS
        temp(mu, lambda) = S(mu, nu)*D(nu, lambda); //SD
        comm(mu, lambda) += -1.0*temp(mu, nu)*F(nu, lambda);//FDS - SDF

    } catch (...) {
        failed = true;
    }
    REQUIRE(!failed);
}

TEST_CASE("SCF GuessDensity declarations") {
    bool failed = false;
    try {
        using tensor_type = tamm::Tensor<double>;
        //tamm::TiledIndexSpace MOs = rv.C.get_spaces()[1];
        //tamm::TiledIndexSpace AOs = rv.C.get_spaces()[0];
        tamm::TiledIndexSpace MOs ;
        tamm::TiledIndexSpace AOs ;
        tamm::TiledIndexLabel i, mu, nu;
        std::tie(mu, nu) = AOs.labels<2>("all");
        std::tie(i) = MOs.labels<1>("all");
    } catch (...) {
        failed = true;
    }
    REQUIRE(!failed);
}

TEST_CASE("SCF JK declarations") {
    bool failed = false;
    try {
        using tensor_type = tamm::Tensor<double>;
        //tamm::TiledIndexSpace Aux = M.get_spaces()[0];
        //tamm::TiledIndexSpace AOs = I.get_spaces()[1];
        //tamm::TiledIndexSpace tMOs = MOs.Cdagger.get_spaces()[0];
        tamm::TiledIndexSpace Aux;
        tamm::TiledIndexSpace AOs;
        tamm::TiledIndexSpace tMOs;
        tamm::TiledIndexLabel P, Q, mu, nu, i;
        std::tie(P, Q) = Aux.labels<2>("all");
        std::tie(mu, nu) = AOs.labels<2>("all");
        std::tie(i) = tMOs.labels<1>("all");

        tensor_type L;
        tensor_type Linv;
        tensor_type Itemp, D, d, J, K;

        //Itemp(Q, i, nu) = MOs.Cdagger(i, mu) * I(Q, mu, nu);
        D(P, i, mu) = Linv(P, Q) * Itemp(Q, i, mu);
        //d(P) = D(P, i, mu) * MOs.Cdagger(i, mu);
        Itemp(Q) = d(P) * Linv(P, Q);
        //J(mu, nu) = Itemp(P) * I(P, mu, nu);
        K(mu, nu) = D(P, i, mu) * D(P, i, nu);

    } catch (...) {
        failed = true;
    }
    REQUIRE(!failed);
}
