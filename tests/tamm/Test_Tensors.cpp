// #define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_RUNNER
#include <catch/catch.hpp>

#include "ga-mpi.h"
#include "ga.h"
#include "macdecls.h"
#include "mpi.h"
#include <tamm/tamm.hpp>

using namespace tamm;

using T = double;

ExecutionContext make_execution_context() {
    ProcGroup pg{GA_MPI_Comm()};
    auto* pMM             = MemoryManagerLocal::create_coll(pg);
    Distribution_NW* dist = new Distribution_NW();
    return ExecutionContext(pg, dist, pMM);
}

void lambda_function(const IndexVector& blockid, span<T> buff) {
    for(size_t i = 0; i < static_cast<size_t>(buff.size()); i++) { buff[i] = 42; }
}

template<typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T>& vec) {
    os << "[";
    for(auto& x : vec) os << x << ",";
    os << "]\n";
    return os;
}

template<typename T>
void print_tensor(Tensor<T>& t) {
    auto lt = t();
    for(auto it : t.loop_nest()) {
        auto blockid   = internal::translate_blockid(it, lt);
        TAMM_SIZE size = t.block_size(blockid);
        std::vector<T> buf(size);
        t.get(blockid, buf);
        std::cout << "block" << blockid;
        for(TAMM_SIZE i = 0; i < size; i++) std::cout << buf[i] << " ";
        std::cout << std::endl;
    }
}

template<typename T>
void check_value(LabeledTensor<T> lt, T val) {
    LabelLoopNest loop_nest{lt.labels()};
    T ref_val = val;
    for(const auto& itval : loop_nest) {
        const IndexVector blockid = internal::translate_blockid(itval, lt);
        size_t size               = lt.tensor().block_size(blockid);
        std::vector<T> buf(size);
        lt.tensor().get(blockid, buf);
        if(lt.tensor().is_non_zero(blockid)) {
            ref_val = val;
        } else {
            ref_val = (T)0;
        }
        for(TAMM_SIZE i = 0; i < size; i++) {
            REQUIRE(std::fabs(buf[i] - ref_val) < 1.0e-10);
        }
    }
}

template<typename T>
void check_value(Tensor<T>& t, T val) {
    check_value(t(), val);
}

template<typename T>
void tensor_contruction(const TiledIndexSpace& T_AO,
                        const TiledIndexSpace& T_MO,
                        const TiledIndexSpace& T_ATOM,
                        const TiledIndexSpace& T_AO_ATOM) {
    TiledIndexLabel A, r, s, mu, mu_A;

    A              = T_ATOM.label("all", 1);
    std::tie(r, s) = T_MO.labels<2>("all");
    mu             = T_AO.label("all", 1);
    mu_A           = T_AO_ATOM.label("all", 1);

    // Tensor Q{T_ATOM, T_MO, T_MO}, C{T_AO,T_MO}, SC{T_AO,T_MO};
    Tensor<T> Q{A, r, s}, C{mu, r}, SC{mu, s};

    Q(A, r, s) = 0.5 * C(mu_A(A), r) * SC(mu_A(A), s);
    Q(A, r, s) += 0.5 * C(mu_A(A), s) * SC(mu_A(A), r);
}

TEST_CASE("Spin Tensor Construction") {
    using T = double;
    IndexSpace SpinIS{range(0, 20),
                      {{"occ", {range(0, 10)}}, {"virt", {range(10, 20)}}},
                      {{Spin{1}, {range(0, 5), range(10, 15)}},
                       {Spin{2}, {range(5, 10), range(15, 20)}}}};

    IndexSpace IS{range(0, 20)};

    TiledIndexSpace SpinTIS{SpinIS, 5};
    TiledIndexSpace TIS{IS, 5};

    std::vector<SpinPosition> spin_mask_2D{SpinPosition::lower,
                                           SpinPosition::upper};

    TiledIndexLabel i, j, k, l;
    std::tie(i, j) = SpinTIS.labels<2>("all");
    std::tie(k, l) = TIS.labels<2>("all");

    bool failed = false;
    try {
        TiledIndexSpaceVec t_spaces{SpinTIS, SpinTIS};
        Tensor<T> tensor{t_spaces, spin_mask_2D};
    } catch(const std::string& e) {
        std::cerr << e << '\n';
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        IndexLabelVec t_lbls{i, j};
        Tensor<T> tensor{t_lbls, spin_mask_2D};
    } catch(const std::string& e) {
        std::cerr << e << '\n';
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        TiledIndexSpaceVec t_spaces{TIS, TIS};

        Tensor<T> tensor{t_spaces, spin_mask_2D};
    } catch(const std::string& e) {
        std::cerr << e << '\n';
        failed = true;
    }
    REQUIRE(!failed);

    {
        REQUIRE((SpinTIS.spin(0) == Spin{1}));
        REQUIRE((SpinTIS.spin(1) == Spin{2}));
        REQUIRE((SpinTIS.spin(2) == Spin{1}));
        REQUIRE((SpinTIS.spin(3) == Spin{2}));

        REQUIRE((SpinTIS("occ").spin(0) == Spin{1}));
        REQUIRE((SpinTIS("occ").spin(1) == Spin{2}));

        REQUIRE((SpinTIS("virt").spin(0) == Spin{1}));
        REQUIRE((SpinTIS("virt").spin(1) == Spin{2}));
    }

    TiledIndexSpace tis_3{SpinIS, 3};

    {
        REQUIRE((tis_3.spin(0) == Spin{1}));
        REQUIRE((tis_3.spin(1) == Spin{1}));
        REQUIRE((tis_3.spin(2) == Spin{2}));
        REQUIRE((tis_3.spin(3) == Spin{2}));
        REQUIRE((tis_3.spin(4) == Spin{1}));
        REQUIRE((tis_3.spin(5) == Spin{1}));
        REQUIRE((tis_3.spin(6) == Spin{2}));
        REQUIRE((tis_3.spin(7) == Spin{2}));

        REQUIRE((tis_3("occ").spin(0) == Spin{1}));
        REQUIRE((tis_3("occ").spin(1) == Spin{1}));
        REQUIRE((tis_3("occ").spin(2) == Spin{2}));
        REQUIRE((tis_3("occ").spin(3) == Spin{2}));

        REQUIRE((tis_3("virt").spin(0) == Spin{1}));
        REQUIRE((tis_3("virt").spin(1) == Spin{1}));
        REQUIRE((tis_3("virt").spin(2) == Spin{2}));
        REQUIRE((tis_3("virt").spin(3) == Spin{2}));
    }

    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};

    failed = false;
    try {
        TiledIndexSpaceVec t_spaces{tis_3, tis_3};
        Tensor<T> tensor{t_spaces, spin_mask_2D};
        tensor.allocate(ec);
        Scheduler{ec}(tensor() = 42).execute();
        check_value(tensor, (T)42);
        tensor.deallocate();
    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        TiledIndexSpaceVec t_spaces{tis_3("occ"), tis_3("virt")};
        Tensor<T> tensor{t_spaces, spin_mask_2D};
        tensor.allocate(ec);

        Scheduler{ec}(tensor() = 42).execute();
        check_value(tensor, (T)42);
        tensor.deallocate();
    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        TiledIndexSpaceVec t_spaces{tis_3, tis_3};
        Tensor<T> T1{t_spaces, spin_mask_2D};
        Tensor<T> T2{t_spaces, spin_mask_2D};
        T1.allocate(ec);
        T2.allocate(ec);

        Scheduler{ec}(T2() = 3)(T1() = T2()).execute();
        check_value(T2, (T)3);
        check_value(T1, (T)3);

        T1.deallocate();
        T2.deallocate();
    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        TiledIndexSpaceVec t_spaces{tis_3, tis_3};
        Tensor<T> T1{t_spaces, {1, 1}};
        Tensor<T> T2{t_spaces, {1, 1}};
        T1.allocate(ec);
        T2.allocate(ec);

        Scheduler{ec}(T1() = 42)(T2() = 3)(T1() += T2()).execute();
        check_value(T2, (T)3);
        check_value(T1, (T)45);

        T1.deallocate();
        T2.deallocate();
    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        TiledIndexSpaceVec t_spaces{tis_3, tis_3};
        Tensor<T> T1{t_spaces, {1, 1}};
        Tensor<T> T2{t_spaces, {1, 1}};
        T1.allocate(ec);
        T2.allocate(ec);

        Scheduler{ec}(T1() = 42)(T2() = 3)(T1() += 2 * T2()).execute();
        check_value(T2, (T)3);
        check_value(T1, (T)48);

        T1.deallocate();
        T2.deallocate();
    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        TiledIndexSpaceVec t_spaces{tis_3, tis_3};
        Tensor<T> T1{t_spaces, {1, 1}};
        Tensor<T> T2{t_spaces, {1, 1}};
        Tensor<T> T3{t_spaces, {1, 1}};

        T1.allocate(ec);
        T2.allocate(ec);
        T3.allocate(ec);

        Tensor<T> T4 = T3;

        Scheduler{ec}(T1() = 42)(T2() = 3)(T3() = 4)(T1() += T4() * T2())
          .execute();
        check_value(T3, (T)4);
        check_value(T2, (T)3);
        check_value(T1, (T)54);

        T1.deallocate();
        T2.deallocate();
        T3.deallocate();
    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        auto lambda = [&](const IndexVector& blockid, span<T> buff) {
            for(size_t i = 0; i < static_cast<size_t>(buff.size()); i++) { buff[i] = 42; }
        };
        TiledIndexSpaceVec t_spaces{TIS, TIS};
        Tensor<T> t{t_spaces, lambda};

        auto lt = t();
        for(auto it : t.loop_nest()) {
            auto blockid   = internal::translate_blockid(it, lt);
            TAMM_SIZE size = t.block_size(blockid);
            std::vector<T> buf(size);
            t.get(blockid, buf);
            std::cout << "block" << blockid;
            for(TAMM_SIZE i = 0; i < size; i++) std::cout << buf[i] << " ";
            std::cout << std::endl;
        }

    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        auto lambda = [](const IndexVector& blockid, span<T> buff) {
            for(size_t i = 0; i < static_cast<size_t>(buff.size()); i++) { buff[i] = 42; }
        };
        // TiledIndexSpaceVec t_spaces{TIS, TIS};
        Tensor<T> S{{TIS, TIS}, lambda};
        Tensor<T> T1{{TIS, TIS}};

        T1.allocate(ec);

        Scheduler{ec}(T1() = 0)(T1() += 2 * S()).execute();

        check_value(T1, (T)84);

    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        Tensor<T> S{{TIS, TIS}, lambda_function};
        Tensor<T> T1{{TIS, TIS}};

        T1.allocate(ec);

        Scheduler{ec}(T1() = 0)(T1() += 2 * S()).execute();

        check_value(T1, (T)84);

    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        std::vector<Tensor<T>> x1(5);
        std::vector<Tensor<T>> x2(5);
        for(int i = 0; i < 5; i++) {
            x1[i] = Tensor<T>{TIS, TIS};
            x2[i] = Tensor<T>{TIS, TIS};
            Tensor<T>::allocate(ec, x1[i], x2[i]);
        }

        auto deallocate_vtensors = [&](auto&&... vecx) {
            //(std::for_each(vecx.begin(), vecx.end(),
            // std::mem_fun(&Tensor<T>::deallocate)), ...);
            //(std::for_each(vecx.begin(), vecx.end(), Tensor<T>::deallocate),
            //...);
        };
        deallocate_vtensors(x1, x2);
    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        IndexSpace MO_IS{range(0, 7)};
        TiledIndexSpace MO{MO_IS, {1, 1, 3, 1, 1}};

        IndexSpace MO_IS2{range(0, 7)};
        TiledIndexSpace MO2{MO_IS2, {1, 1, 3, 1, 1}};

        Tensor<T> pT{MO, MO};
        Tensor<T> pV{MO2, MO2};

        pT.allocate(ec);
        pV.allocate(ec);

        auto tis_list = pT.tiled_index_spaces();
        Tensor<T> H{tis_list};
        H.allocate(ec);

        auto h_tis = H.tiled_index_spaces();

        Scheduler{ec}(H("mu", "nu") = pT("mu", "nu"))(H("mu", "nu") +=
                                                      pV("mu", "nu"))
          .execute();

    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        IndexSpace IS{range(10)};
        TiledIndexSpace TIS{IS, 2};

        Tensor<T> A{TIS, TIS};
        auto ec = make_execution_context();
        A.allocate(&ec);
    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);

    failed = false;
    try {
        IndexSpace AO_IS{range(10)};
        TiledIndexSpace AO{AO_IS, 2};
        IndexSpace MO_IS{range(10)};
        TiledIndexSpace MO{MO_IS, 2};

        Tensor<T> C{AO, MO};
        auto ec_temp = make_execution_context();
        C.allocate(&ec_temp);
        // Scheduler{&ec}.allocate(C)
        //     (C() = 42.0).execute();

        const auto AOs = C.tiled_index_spaces()[0];
        const auto MOs = C.tiled_index_spaces()[1];
        auto [mu, nu]  = AOs.labels<2>("all");

        // TODO: Take the slice of C that is for the occupied orbitals
        auto [p] = MOs.labels<1>("all");
        Tensor<T> rho{AOs, AOs};

        tamm::ProcGroup pg{GA_MPI_Comm()};
        auto* pMM = tamm::MemoryManagerLocal::create_coll(pg);
        tamm::Distribution_NW dist;
        tamm::ExecutionContext ec(pg, &dist, pMM);
        tamm::Scheduler sch{&ec};

        sch.allocate(rho)(rho() = 0)(rho(mu, nu) += C(mu, p) * C(nu, p))
          .execute();

    } catch(const std::string& e) {
        std::cerr << e << std::endl;
        failed = true;
    }
    REQUIRE(!failed);
}

TEST_CASE("Hash Based Equality and Compatibility Check") {
    

    IndexSpace is1{range(0, 20),
                   {{"occ", {range(0, 10)}}, {"virt", {range(10, 20)}}}};
    IndexSpace is2{range(0, 10)};
    IndexSpace is1_occ = is1("occ");

    TiledIndexSpace tis1{is1};
    TiledIndexSpace tis2{is2};
    TiledIndexSpace tis3{is1_occ};
    TiledIndexSpace sub_tis1{tis1, range(0, 10)};

    REQUIRE(tis2 == tis3);
    REQUIRE(tis2 == tis1("occ"));
    REQUIRE(tis3 == tis1("occ"));
    REQUIRE(tis1 != tis2);
    REQUIRE(tis1 != tis3);
    REQUIRE(tis2 != tis1("virt"));
    REQUIRE(tis3 != tis1("virt"));

    // sub-TIS vs TIS from same IS
    REQUIRE(sub_tis1 == tis2);
    REQUIRE(sub_tis1 == tis3);
    REQUIRE(sub_tis1 == tis1("occ"));
    REQUIRE(sub_tis1 != tis1);
    REQUIRE(sub_tis1 != tis1("virt"));


    REQUIRE(sub_tis1.is_compatible_with(tis1));
    REQUIRE(sub_tis1.is_compatible_with(tis1("occ")));
    REQUIRE(sub_tis1.is_compatible_with(tis2));
    REQUIRE(sub_tis1.is_compatible_with(tis3));
    REQUIRE(!sub_tis1.is_compatible_with(tis1("virt")));
    
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    GA_Initialize();
    MA_init(MT_DBL, 8000000, 20000000);

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int res = Catch::Session().run(argc, argv);
    GA_Terminate();
    MPI_Finalize();

    return res;
}