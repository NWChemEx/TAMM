#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#include "ga-mpi.h"
#include "ga.h"
#include "macdecls.h"
#include "mpi.h"
#include "tamm/tamm.hpp"
#include "tamm/utils.hpp"

using namespace tamm;


template<typename T>
void check_value(const LabeledTensor<T>& lt, T val) {
    LabelLoopNest loop_nest{lt.labels()};

    for(const auto& itval : loop_nest) {
        const IndexVector blockid = internal::translate_blockid(itval, lt);
        size_t size               = lt.tensor().block_size(blockid);
        std::vector<T> buf(size);
        lt.tensor().get(blockid, buf);
        for(TAMM_SIZE i = 0; i < size; i++) {
            REQUIRE(std::fabs(buf[i] - val) < 1.0e-10);
        }
    }
}

template<typename T>
void check_value(const Tensor<T>& t, T val) {
    check_value(t(), val);
}

template<typename T>
void check_and_print(const Tensor<T>& tensor, T val) {
    check_value(tensor, val);
    print_tensor(tensor);
}

TEST_CASE("Zero-dimensional tensor with double") {
    bool failed = false;
    // IndexSpace is {range(10)};
    // TiledIndexSpace tis{is};
    Tensor<double> T1{};
    try {
        T1() = double{0};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
}

TEST_CASE("Zero-dimensional tensor with int") {
    bool failed = false;
    // IndexSpace is {range(10)};
    // TiledIndexSpace tis{is};
    Tensor<double> T1{};
    try {
        T1() = int{0};
    } catch(...) { failed = true; }
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
    IndexSpace is{range(10)};
    TiledIndexSpace tis{is};
    Tensor<double> T1{tis};
    try {
        T1() = double{7};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    try {
        T1("i") = double{0};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    TiledIndexLabel i = tis.label("all");
    try {
        T1(i) = double{5};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
}

TEST_CASE("One-dimensional tensor with int") {
    bool failed = false;
    IndexSpace is{range(10)};
    TiledIndexSpace tis{is};
    Tensor<double> T1{tis};
    try {
        T1() = int{7};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    try {
        T1("i") = int{0};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    TiledIndexLabel i = tis.label("all");
    try {
        T1(i) = int{5};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
}

TEST_CASE("Two-dimensional tensor") {
    bool failed = false;
    IndexSpace is{range(10)};
    TiledIndexSpace tis{is};
    Tensor<double> T1{tis, tis};
    TiledIndexLabel i, j;
    std::tie(i, j) = tis.labels<2>("all");

    try {
        T1() = double{8};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    try {
        T1(i, j) = double{5};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    try {
        T1(i, "j") = double{5};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    try {
        T1(i, "i") = double{5};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    try {
        T1("x", "x") = double{5};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    try {
        T1(i, i) = double{5};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    // invalid operations: should fail

    // invalid number of labels
    try {
        T1(i) = double{8};
    } catch(...) { failed = true; }
    REQUIRE(failed);
    failed = false;

    // invalid number of labels
    try {
        T1("x") = double{8};
    } catch(...) { failed = true; }
    REQUIRE(failed);
    failed = false;

    // invalid labels
    try {
        T1(i, i(j)) = double{8};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    // invalid labels
    try {
        T1(i, j(i)) = double{8};
    } catch(...) { failed = true; }
    REQUIRE(!failed);
    failed = false;

    // invalid number of labels
    try {
        T1(i, j, "x") = double{8};
    } catch(...) { failed = true; }
    REQUIRE(failed);
    failed = false;
}

TEST_CASE("SCF Commutator declarations") {
    bool failed = false;
    try {
        using tensor_type = tamm::Tensor<double>;
        using space_type  = tamm::TiledIndexSpace;
        using index_type  = tamm::TiledIndexLabel;

        ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
        ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};

        IndexSpace is{range(10)};
        space_type tis{is};

        tensor_type comm{tis, tis}, temp{tis, tis}, F{tis, tis}, D{tis, tis},
          S{tis, tis};
        index_type mu, nu, lambda;

        std::tie(mu, nu, lambda) = tis.labels<3>("all");

        tensor_type::allocate(ec, comm, temp, F, D, S);

        /*
        temp(mu, lambda) = F(mu,nu)*D(nu, lambda); //FD
        comm(mu, lambda) = temp(mu, nu)*S(nu, lambda); //FDS
        temp(mu, lambda) = S(mu, nu)*D(nu, lambda); //SD
        comm(mu, lambda) += -1.0*temp(mu, nu)*F(nu, lambda);//FDS - SDF
         */

        Scheduler{*ec}(temp(mu, lambda) += F(mu, nu) * D(nu, lambda)) // FD
          (comm(mu, lambda) += temp(mu, nu) * S(nu, lambda))         // FDS
          (temp(mu, lambda) += S(mu, nu) * D(nu, lambda))            // SD
          (comm(mu, lambda) += -1.0 * temp(mu, nu) * F(nu, lambda)) // FDS - SDF
            .execute();

        tensor_type::deallocate(comm, temp, F, D, S);

        delete ec;
    } catch(...) { failed = true; }
    REQUIRE(!failed);
}

TEST_CASE("SCF GuessDensity declarations") {
    bool failed = false;
    try {
        // using tensor_type = tamm::Tensor<double>;
        // tamm::TiledIndexSpace MOs = rv.C.get_spaces()[1];
        // tamm::TiledIndexSpace AOs = rv.C.get_spaces()[0];
        IndexSpace mo_is{range(10)};
        IndexSpace ao_is{range(10, 20)};

        tamm::TiledIndexSpace MOs{mo_is};
        tamm::TiledIndexSpace AOs{ao_is};
        tamm::TiledIndexLabel i, mu, nu;
        std::tie(mu, nu) = AOs.labels<2>("all");
        std::tie(i)      = MOs.labels<1>("all");
    } catch(...) { failed = true; }
    REQUIRE(!failed);
}

TEST_CASE("SCF JK declarations") {
    bool failed = false;
    try {
        using tensor_type = tamm::Tensor<double>;

        ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
        ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

        IndexSpace is{range(10)};
        tamm::TiledIndexSpace tis{is};

        // tamm::TiledIndexSpace Aux = M.get_spaces()[0];
        // tamm::TiledIndexSpace AOs = I.get_spaces()[1];
        // tamm::TiledIndexSpace tMOs = MOs.Cdagger.get_spaces()[0];
        tamm::TiledIndexSpace Aux{is};
        tamm::TiledIndexSpace AOs{is};
        tamm::TiledIndexSpace tMOs{is};
        tamm::TiledIndexLabel P, Q, mu, nu, i;
        std::tie(P, Q)   = Aux.labels<2>("all");
        std::tie(mu, nu) = AOs.labels<2>("all");
        std::tie(i)      = tMOs.labels<1>("all");

        tensor_type L{tis, tis};
        tensor_type Linv{Aux, Aux};
        tensor_type Itemp{Aux, tMOs, AOs};
        tensor_type D{Aux, tMOs, AOs};
        tensor_type d{Aux};
        tensor_type J{tis, tis};
        tensor_type K{AOs, AOs};

        tensor_type::allocate(&ec, L, Linv, Itemp, D, d, J, K);

        // // Itemp(Q, i, nu) = MOs.Cdagger(i, mu) * I(Q, mu, nu);
        Scheduler{ec}
            (D() = 1.0)
            (Linv() = 2.0)
            (Itemp() = 3.0)
            (D() = 4.0)
            // (D(P, i, mu) += Linv(P, Q) * Itemp(Q, i, mu))
          // d(P) = D(P, i, mu) * MOs.Cdagger(i, mu);
          //@TODO cannot use itemp this way
          //(Itemp(Q) = d(P) * Linv(P, Q))
          // J(mu, nu) = Itemp(P) * I(P, mu, nu);
        //  (K(mu, nu) += D(P, i, mu) * D(P, i, nu))
            .execute();

        // tensor_type::deallocate(L, Linv, Itemp, D, d, J, K);
        // MemoryManagerGA::destroy_coll(mgr);
        // delete ec;

    } catch(...) { failed = true; }
    REQUIRE(!failed);
}
#if 0
TEST_CASE("CCSD T2") {
    bool failed = false;

    try {
        using T = double;

        ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
        ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};

        IndexSpace MO_IS{range(0, 14),
                         {{"occ", {range(0, 10)}}, {"virt", {range(10, 14)}}}};

        TiledIndexSpace MO{MO_IS, 10};

        TiledIndexSpace O = MO("occ");
        TiledIndexSpace V = MO("virt");
        TiledIndexSpace N = MO("all");

        Tensor<T> d_f1{N, N};
        Tensor<T> d_r1{O, O};
        Tensor<T>::allocate(ec, d_r1, d_f1);

        TiledIndexLabel h1, h2, h3, h4;
        TiledIndexLabel p1, p2, p3, p4;
        std::tie(h1, h2, h3, h4) = MO.labels<4>("occ");
        std::tie(p1, p2, p3, p4) = MO.labels<4>("virt");

        Scheduler{*ec}(d_r1() = 0).execute();

        block_for(*ec, d_f1(), [&](IndexVector it) {
            Tensor<T> tensor     = d_f1().tensor();
            const TAMM_SIZE size = tensor.block_size(it);

            std::vector<T> buf(size);

            // const int ndim = 2;
            // std::array<int, ndim> block_offset;
            // auto& tiss      = tensor.tiled_index_spaces();
            auto block_dims = tensor.block_dims(it);
            // for(auto i = 0; i < ndim; i++) {
            //     block_offset[i] = tiss[i].tile_offset(it[i]);
            // }
            auto block_offset = tensor.block_offsets(it);

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

        Scheduler{*ec}(d_r1(h1, h2) = d_f1(h1, h2)).execute();

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

TEST_CASE("Tensor operations on named subspaces") {
    IndexSpace is_named{range(0, 14),
                        {{"occ", {range(0, 10)}},
                         {"virt", {range(10, 14)}},
                         {"alpha", {range(0, 5), range(10, 12)}},
                         {"beta", {range(5, 10), range(12, 14)}}}};

    TiledIndexSpace tis_s{is_named, 5};
    TiledIndexSpace tis_l{is_named, {5, 5, 2, 2}};

    std::cerr << "Tiling check" << std::endl;
    REQUIRE(tis_s.num_tiles() == tis_l.num_tiles());
    for(const auto& tile_id : tis_s) {
        REQUIRE(tis_s.tile_size(tile_id) == tis_l.tile_size(tile_id));
    }

    const auto& N     = tis_s("all");
    const auto& O     = tis_s("occ");
    const auto& V     = tis_s("virt");
    const auto& alpha = tis_s("alpha");
    const auto& beta  = tis_s("beta");

    std::cerr << "Compatibility check" << std::endl;

    REQUIRE(N.is_compatible_with(N));
    REQUIRE(O.is_compatible_with(O));
    REQUIRE(V.is_compatible_with(V));
    REQUIRE(alpha.is_compatible_with(alpha));
    REQUIRE(beta.is_compatible_with(beta));

    REQUIRE(O.is_compatible_with(N));
    REQUIRE(V.is_compatible_with(N));
    REQUIRE(alpha.is_compatible_with(N));
    REQUIRE(beta.is_compatible_with(N));

    REQUIRE(!O.is_compatible_with(V));
    REQUIRE(!O.is_compatible_with(alpha));
    REQUIRE(!O.is_compatible_with(beta));

    REQUIRE(!V.is_compatible_with(O));
    REQUIRE(!V.is_compatible_with(alpha));
    REQUIRE(!V.is_compatible_with(beta));

    REQUIRE(!alpha.is_compatible_with(O));
    REQUIRE(!alpha.is_compatible_with(V));
    REQUIRE(!alpha.is_compatible_with(beta));

    REQUIRE(!beta.is_compatible_with(O));
    REQUIRE(!beta.is_compatible_with(V));
    REQUIRE(!beta.is_compatible_with(alpha));

    std::cerr << "Construct tensors using different tiled index spaces"
              << std::endl;
    using T = double;
    Tensor<T> T1{N, N};
    Tensor<T> T2{O, O};
    Tensor<T> T3{V, V};
    Tensor<T> T4{O, V};
    Tensor<T> T5{V, O};
    Tensor<T> T6{O, N};
    Tensor<T> T7{V, N};

    std::cerr << "Allocate and deallocate tensors" << std::endl;

    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};
    {
        bool failed = false;
        try {
            Tensor<T>::allocate(ec, T1, T2, T3, T4, T5, T6, T7);
            Tensor<T>::deallocate(T1, T2, T3, T4, T5, T6, T7);
        } catch(std::string& e) {
            std::cerr << "Caught exception: " << e << "\n";
            failed = true;
        }
        REQUIRE(!failed);
    }

    //Test with default ExecutionContext constructor
    //FIXME: We do not allow allocating the same tensor after deallocating it
    {
        bool failed = false;

        using T = double;
        Tensor<T> T1{N, N};
        Tensor<T> T2{O, O};
        Tensor<T> T3{V, V};
        Tensor<T> T4{O, V};
        Tensor<T> T5{V, O};
        Tensor<T> T6{O, N};
        Tensor<T> T7{V, N};

        try {
            ExecutionContext temp_ec;
            temp_ec.set_pg(pg);
            temp_ec.set_distribution(&distribution);
            temp_ec.set_memory_manager(mgr);

            Tensor<T>::allocate(&temp_ec, T1, T2, T3, T4, T5, T6, T7);
            Tensor<T>::deallocate(T1, T2, T3, T4, T5, T6, T7);
        } catch(std::string& e) {
            std::cerr << "Caught exception: " << e << "\n";
            failed = true;
        }
        REQUIRE(!failed);
    }

    std::cerr << "Tensor allocate/deallocate with Tensor member functions" << std::endl;
    {
        bool failed = false;
        using T = double;
        Tensor<T> T1{N, N};
        Tensor<T> T2{O, O};
        Tensor<T> T3{V, V};
        Tensor<T> T4{O, V};
        Tensor<T> T5{V, O};
        Tensor<T> T6{O, N};
        Tensor<T> T7{V, N};
        
        try {
            std::vector<Tensor<T>> tensor_vec{T1, T2, T3, T4, T5, T6, T7};

            for(auto tensor : tensor_vec) {
                tensor.allocate(ec);
            }
            
            for(auto tensor : tensor_vec) {
                tensor.deallocate();
            }

        } catch(std::string& e) {
            std::cerr << "Caught exception: " << e << "\n";
            failed = true;
        }
        REQUIRE(!failed);
    }

    std::cerr << "Ops on the same tiled index space" << std::endl;

    {
        bool failed = false;

        using T = double;
        Tensor<T> T1{N, N};
        Tensor<T> T2{O, O};
        Tensor<T> T3{V, V};
        Tensor<T> T4{O, V};
        Tensor<T> T5{V, O};
        Tensor<T> T6{O, N};
        Tensor<T> T7{V, N};


        try {
            Scheduler{*ec}
              .allocate(T1, T2, T3, T4, T5, T6, T7)
              (T1() = 1)(T2() = 2)
              (T3() = 3)(T4() = 4)
              (T5() = 5)(T6() = 6)
              (T7() = 7)
              .execute();

            check_and_print(T1, 1.0);
            check_and_print(T2, 2.0);
            check_and_print(T3, 3.0);
            check_and_print(T4, 4.0);
            check_and_print(T5, 5.0);
            check_and_print(T6, 6.0);
            check_and_print(T7, 7.0);

            Tensor<T>::deallocate(T1, T2, T3, T4, T5, T6, T7);
        } catch(const std::string& e) {
            std::cerr << "Caught exception: " << e << "\n";
            failed = true;
        }
        REQUIRE(!failed);
    }

    std::cerr << "Check operation on compatible tiled index spaces"
              << std::endl;

    {
        bool failed = false;

        using T = double;
        Tensor<T> T1{N, N};
        Tensor<T> T2{O, O};
        Tensor<T> T3{V, V};
        Tensor<T> T4{O, V};
        Tensor<T> T5{V, O};
        Tensor<T> T6{O, N};
        Tensor<T> T7{V, N};

        try {
            Scheduler{*ec}
              .allocate(T1, T2, T3, T4, T5, T6, T7)
              (T1() = 1)(T2() = 2)
              (T3() = 3)(T4() = 4)
              (T5() = 5)(T6() = 6)
              (T7() = 7)
              .execute();
  

            TiledIndexLabel i, j, p, o, k, l;

            std::tie(i,j) = N.labels<2>("all");
            std::tie(p,o) = N.labels<2>("occ");
            std::tie(k,l) = N.labels<2>("virt");
            
            Scheduler{*ec}
              (T1(p,o) += T2(p,o))
              (T6(p,o) += T2(p,o))
              (T7(k,l) += T3(k,l))
              .execute();

            Tensor<T>::deallocate(T1, T2, T3, T4, T5, T6, T7);
        } catch(const std::string& e) {
            std::cerr << "Caught exception: " << e << "\n";
            failed = true;
        }
        REQUIRE(!failed);
    }

    std::cerr << "Check operation on incompatible tiled index spaces"
              << std::endl;

    {
        bool failed = false;

        using T = double;
        Tensor<T> T1{N, N};
        Tensor<T> T2{O, O};
        Tensor<T> T3{V, V};
        Tensor<T> T4{O, V};
        Tensor<T> T5{V, O};
        Tensor<T> T6{O, N};
        Tensor<T> T7{V, N};

        try {
            Scheduler{*ec}
              .allocate(T1, T2, T3, T4, T5, T6, T7)
              (T1() = 1)(T2() = 2)
              (T3() = 3)(T4() = 4)
              (T5() = 5)(T6() = 6)
              (T7() = 7)
              .execute();
  

            TiledIndexLabel i, j, p, o, k, l;

            std::tie(i,j) = N.labels<2>("all");
            std::tie(p,o) = N.labels<2>("occ");
            std::tie(k,l) = N.labels<2>("virt");
            
            Scheduler{*ec}
              (T2(p,o) = T3(k,l))
              (T2(p,o) += T3(k,l))
              .execute();

            print_tensor(T2);

            Tensor<T>::deallocate(T1, T2, T3, T4, T5, T6, T7);
        } catch(const std::string& e) {
            std::cerr << "Caught exception: " << e << "\n";
            failed = true;
        }
        REQUIRE(!failed);
    }
}

#endif


int main(int argc, char* argv[]) {

    tamm::initialize(argc, argv);

    doctest::Context context(argc, argv);

    int res = context.run();

    tamm::finalize();

    return res;
}
