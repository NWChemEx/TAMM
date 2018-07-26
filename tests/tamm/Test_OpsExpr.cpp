#define CATCH_CONFIG_RUNNER

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

        ProcGroup pg{GA_MPI_Comm()};
        auto mgr = MemoryManagerGA::create_coll(pg);
        Distribution_NW distribution;
        ExecutionContext *ec = new ExecutionContext{pg,&distribution,mgr};

        IndexSpace is{range(10)};
        space_type tis{is};

        tensor_type comm{tis, tis}, temp{tis, tis}, F{tis, tis}, D{tis, tis}, S{tis, tis};
        index_type mu, nu, lambda;

        std::tie(mu,nu,lambda) = tis.labels<3>("all");

        tensor_type::allocate(ec, comm, temp, F, D, S);

        /*
        temp(mu, lambda) = F(mu,nu)*D(nu, lambda); //FD
        comm(mu, lambda) = temp(mu, nu)*S(nu, lambda); //FDS
        temp(mu, lambda) = S(mu, nu)*D(nu, lambda); //SD
        comm(mu, lambda) += -1.0*temp(mu, nu)*F(nu, lambda);//FDS - SDF
         */

        Scheduler{ec}
            (temp(mu, lambda) = F(mu,nu)*D(nu, lambda)) //FD
            (comm(mu, lambda) = temp(mu, nu)*S(nu, lambda)) //FDS
            (temp(mu, lambda) = S(mu, nu)*D(nu, lambda)) //SD
            (comm(mu, lambda) += -1.0*temp(mu, nu)*F(nu, lambda))//FDS - SDF
            .execute();


        tensor_type::deallocate(comm, temp, F, D, S);
        delete ec;
    } catch (...) {
        failed = true;
    }
    REQUIRE(!failed);
}

TEST_CASE("SCF GuessDensity declarations") {
    bool failed = false;
    try {
        //using tensor_type = tamm::Tensor<double>;
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

        ProcGroup pg{GA_MPI_Comm()};
        auto mgr = MemoryManagerGA::create_coll(pg);
        Distribution_NW distribution;
        ExecutionContext *ec = new ExecutionContext{pg,&distribution,mgr};

        IndexSpace is{range(10)};
        tamm::TiledIndexSpace tis{is};

        //tamm::TiledIndexSpace Aux = M.get_spaces()[0];
        //tamm::TiledIndexSpace AOs = I.get_spaces()[1];
        //tamm::TiledIndexSpace tMOs = MOs.Cdagger.get_spaces()[0];
        tamm::TiledIndexSpace Aux{is};
        tamm::TiledIndexSpace AOs{is};
        tamm::TiledIndexSpace tMOs{is};
        tamm::TiledIndexLabel P, Q, mu, nu, i;
        std::tie(P, Q) = Aux.labels<2>("all");
        std::tie(mu, nu) = AOs.labels<2>("all");
        std::tie(i) = tMOs.labels<1>("all");

        tensor_type L{tis, tis};
        tensor_type Linv{tis, tis};
        tensor_type Itemp{tis}, D{tis, tis, tis}, d{tis}, J{tis, tis}, K{tis, tis};

        tensor_type::allocate(ec, L, Linv, Itemp, D, d, J, K);

        //Itemp(Q, i, nu) = MOs.Cdagger(i, mu) * I(Q, mu, nu);
        D(P, i, mu) = Linv(P, Q) * Itemp(Q, i, mu);
        //d(P) = D(P, i, mu) * MOs.Cdagger(i, mu);
        Itemp(Q) = d(P) * Linv(P, Q);
        //J(mu, nu) = Itemp(P) * I(P, mu, nu);
        K(mu, nu) = D(P, i, mu) * D(P, i, nu);

        tensor_type::deallocate(L, Linv, Itemp, D, d, J, K);
        delete ec;
    } catch (...) {
        failed = true;
    }
    REQUIRE(!failed);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc,&argv);
    GA_Initialize();
    MA_init(MT_DBL, 8000000, 20000000);
    
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int res = Catch::Session().run(argc, argv);
    GA_Terminate();
    MPI_Finalize();

    return res;
}
