#define CATCH_CONFIG_RUNNER

#include "catch/catch.hpp"
#include "ga.h"
#include "mpi.h"
#include "macdecls.h"
#include "ga-mpi.h"
#include "tamm/tamm.hpp"

using namespace tamm;

template<typename T>
std::ostream& operator << (std::ostream &os, std::vector<T>& vec){
    os << "[";
    for(auto &x: vec)
        os << x << ",";
    os << "]\n";
    return os;
}

template<typename T>
void print_tensor(Tensor<T> &t){
    for (auto it: t.loop_nest())
    {
        TAMM_SIZE size = t.block_size(it);
        std::vector<T> buf(size);
        t.get(it, buf);
        std::cout << "block" << it;
        for (TAMM_SIZE i = 0; i < size;i++)
         std::cout << buf[i] << " ";
        std::cout << std::endl;
    }
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
            (temp(mu, lambda) += F(mu,nu)*D(nu, lambda)) //FD
            (comm(mu, lambda) += temp(mu, nu)*S(nu, lambda)) //FDS
            (temp(mu, lambda) += S(mu, nu)*D(nu, lambda)) //SD
            (comm(mu, lambda) += -1.0*temp(mu, nu)*F(nu, lambda))//FDS - SDF
            .execute();


        tensor_type::deallocate(comm, temp, F, D, S);
        MemoryManagerGA::destroy_coll(mgr);

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
        IndexSpace mo_is{range(10)};
        IndexSpace ao_is{range(10,20)};

        tamm::TiledIndexSpace MOs{mo_is};
        tamm::TiledIndexSpace AOs{ao_is};
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
        tensor_type Linv{Aux, Aux};
        tensor_type Itemp{Aux, tMOs, AOs};
        tensor_type D{Aux, tMOs, AOs};
        tensor_type d{Aux};
        tensor_type J{tis, tis};
        tensor_type K{AOs, AOs};

        tensor_type::allocate(ec, L, Linv, Itemp, D, d, J, K);

        //Itemp(Q, i, nu) = MOs.Cdagger(i, mu) * I(Q, mu, nu);
        Scheduler{ec}
        (D(P, i, mu) += Linv(P, Q) * Itemp(Q, i, mu))
        //d(P) = D(P, i, mu) * MOs.Cdagger(i, mu);
        //@TODO cannot use itemp this way
        //(Itemp(Q) = d(P) * Linv(P, Q))
        //J(mu, nu) = Itemp(P) * I(P, mu, nu);
        (K(mu, nu) += D(P, i, mu) * D(P, i, nu))
        .execute();

        tensor_type::deallocate(L, Linv, Itemp, D, d, J, K);
        MemoryManagerGA::destroy_coll(mgr);
        delete ec;

    } catch (...) {
        failed = true;
    }
    REQUIRE(!failed);
    
}

TEST_CASE("CCSD T2") {

    bool failed = false;

    try {
        using T     = double;

        ProcGroup pg{GA_MPI_Comm()};
        auto mgr = MemoryManagerGA::create_coll(pg);
        Distribution_NW distribution;
        ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};

        IndexSpace MO_IS{range(0, 14),
                        {
                          {"occ", {range(0, 10)}},                  
                          {"virt", {range(10, 14)}}
                        }};

        TiledIndexSpace MO{MO_IS, 10};

        TiledIndexSpace O = MO("occ");
        TiledIndexSpace V = MO("virt");
        TiledIndexSpace N = MO("all");

        Tensor<T> d_f1{N,N};
        Tensor<T> d_r1{O,O};
        Tensor<T>::allocate(ec, d_r1, d_f1);

        TiledIndexLabel h1, h2,h3,h4;
        TiledIndexLabel p1, p2,p3,p4;
        std::tie(h1,h2,h3,h4) = MO.labels<4>("occ");
        std::tie(p1,p2,p3,p4) = MO.labels<4>("virt");

        Scheduler{ec}(d_r1() = 0).execute();

        block_for(ec->pg(), d_f1(), [&](IndexVector it) {
            Tensor<T> tensor     = d_f1().tensor();
            const TAMM_SIZE size = tensor.block_size(it);

            std::vector<T> buf(size);

            const int ndim = 2;
            std::array<int, ndim> block_offset;
            auto& tiss      = tensor.tiled_index_spaces();
            auto block_dims = tensor.block_dims(it);
            for(auto i = 0; i < ndim; i++) {
                block_offset[i] = tiss[i].tile_offset(it[i]);
            }

            TAMM_SIZE c = 0;
            for(auto i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                double n = rand() % 5;
                for(auto j = block_offset[1];
                    j < block_offset[1] + block_dims[1]; j++, c++) {
                    buf[c] = n + j;
                }
            }

            d_f1.put(it, buf);
        });

        Scheduler{ec}(d_r1(h1,h2) = d_f1(h1,h2)).execute();

        // std::cout << "d_f1\n";
        // print_tensor(d_f1);
        // std::cout << "-----------\nd_r1\n";
        // print_tensor(d_r1);

        Tensor<T>::deallocate(d_r1, d_f1);
        MemoryManagerGA::destroy_coll(mgr);
        delete ec;

    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
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
