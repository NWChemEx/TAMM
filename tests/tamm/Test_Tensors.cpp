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

void lambda_function(const IndexVector& blockid, span<T> buff) {
    for(size_t i = 0; i < static_cast<size_t>(buff.size()); i++) {
        buff[i] = 42;
    }
}

template<size_t last_idx>
void l_func(const IndexVector& blockid, span<T> buf) {
    if(blockid[0] == last_idx || blockid[1] == last_idx) {
        for(auto i = 0U; i < buf.size(); i++) buf[i] = -1;
    } else {
        for(auto i = 0U; i < buf.size(); i++) buf[i] = 0;
    }

    if(blockid[0] == last_idx && blockid[1] == last_idx) {
        for(auto i = 0U; i < buf.size(); i++) buf[i] = 0;
    }
};

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

#if 1
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
    RuntimeEngine re;
    ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr, &re};

    failed = false;
    try {
        TiledIndexSpaceVec t_spaces{tis_3, tis_3};
        Tensor<T> tensor{t_spaces, spin_mask_2D};
        tensor.allocate(ec);
        Scheduler{*ec}(tensor() = 42).execute();
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

        Scheduler{*ec}(tensor() = 42).execute();
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

        Scheduler{*ec}(T2() = 3)(T1() = T2()).execute();
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

        Scheduler{*ec}(T1() = 42)(T2() = 3)(T1() += T2()).execute();
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

        Scheduler{*ec}(T1() = 42)(T2() = 3)(T1() += 2 * T2()).execute();
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

        Scheduler{*ec}(T1() = 42)(T2() = 3)(T3() = 4)(T1() += T4() * T2())
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

        Scheduler{*ec}(T1() = 0)(T1() += 2 * S()).execute();

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

        Scheduler{*ec}(T1() = 0)(T1() += 2 * S()).execute();

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

        Scheduler{*ec}(H("mu", "nu") = pT("mu", "nu"))(H("mu", "nu") +=
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

        ProcGroup pg{GA_MPI_Comm()};
        auto* pMM = tamm::MemoryManagerLocal::create_coll(pg);
        tamm::Distribution_NW dist;
        RuntimeEngine re;
        tamm::ExecutionContext ec(pg, &dist, pMM, &re);
        tamm::Scheduler sch{ec};

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
/*
// Z_i_mu-prime^x = E_mu_v^X * C_i^mu * C_mu-prime^v-prime
// Z_i_mu-prime-i^x-prime-i = (E_mu-i_v-prime-i^x-prime-i * C_i^mu-i) * C_mu-prime-i^v-prime-i
// {X-prime-i} = sum over j in j(i) {X_j}
// {Mu-prime-i} = sum over j in j(i) {Mu-prime_j}
// (i, mu-prime-i | x-prime-i)
TEST_CASE("DLPNO") {

    IndexSpace MU{};
    IndexSpace X{};
    IndexSpace dependent_X{};
    IndexSpace dependent_MU{};

    TiledIndexSpace t_Atom{Atom};
    TiledIndexSpace t_dependent_X{dependent_X};
    TiledIndexSpace t_dependent_MU{dependent_MU};

    Tensor<double> Z{};
    Tensor<double> C{};
    Tensor<double> E{};

    auto ec = make_execution_context();

    auto i = t_Atom.labels<1>("all");
    auto [mu, mu_prime, nu_prime] = t_dependent_MU.labels<4>("all");
    auto x_prime = t_dependent_X.labels<1>("all");

    Scheduler{ec}
    (T() = 0.0)
    (T(i, nu(i), x(i)) += E(i, nu(i), x(i)) * C(i, mu(i)))
    (Z(i, nu(i), x(i)) += T(i, nu(i), x(i)) * C(nu(i), nu_prime(i))
    .execute();

    // Scheduler{ec}
    // (T(i, nu_prime(i), x_prime(i)) += E(mu(i), nu_prime(i), x_prime(i)) * C(i,mu(i))
    // (Z(i, mu_prime(i), x_prime(i)) += T(i, nu_prime(i), x_prime(i)) * C(mu_prime(i), nu_prime(i)))
    // .execute();

}


TEST_CASE("PNO-MP2") {
    // IndexSpace for i, j values (can be different IndexSpaces)
    IndexSpace IS{range(10),
                  {{"occ", {range(0, 5)}},
                   {"virt", {range(5, 10)}}
    }};
    // Dependent IndexSpace for a, b ranges
    // IndexSpace IS_DEP{range(0,6)};
    IndexSpace MOs{range(0, 6),
                   {{"O", {range(0, 3)}},
                   {"V", {range(3, 6)}}
    }};

    auto IS_DEP = MOs("O");
    auto IS_DEP2 = MOs("V");

    // Default tiling for IndexSpace for i, j values (can be different IndexSpaces)
    TiledIndexSpace tIS{IS};      

   
    // Dependency relation between i values and label j
    std::map<IndexVector, IndexSpace> dep_relation_j;

    // Set the dependency for each (i, j) pair and labels a, b
    // here the dependency set on the IS_DEP named subspaces 
    // but it can be done in any way. Assumption is that this  
    // will be passed to the method
    std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
    
    for(const auto& i : IS) {
        if(i%2 == 0){
            dep_relation_j.insert({{i}, IS_DEP});
        }
        else {
            dep_relation_j.insert({{i}, IS_DEP2});
        }

    }

    std::cerr << "(i) -> IndexSpace" << std::endl;
    for(const auto& [key, value] : dep_relation_j) {
        for(const auto& var : key) {
            std::cerr << var << " ";
        }
        std::cerr << "-> { ";

        for(const auto& i : value) {
            std::cerr << i << " ";
        }
        
        std::cerr << "}" << std::endl;

    }

    // Dependent IndexSpace for j
    IndexSpace dep_IS_J{{tIS}, dep_relation_j};
    std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;

    // Dependency relation between (i, j) pairs and labels a, b
    std::map<IndexVector, IndexSpace> dep_relation;
    for(const auto& i : IS) {
        if(i%2 == 0) {
            for(Index j = 0; j < 3; j++) {
                dep_relation.insert({{i, j}, IS_DEP2});
            }
        } else {
            for(Index j = 0; j < 3; j++) {
                dep_relation.insert({{i, j}, IS_DEP});
            }            
        }
    }
    std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;

    std::cerr << "(i, j) -> IndexSpace" << std::endl;
    for(const auto& [key, value] : dep_relation) {
        for(const auto& var : key) {
            std::cerr << var << " ";
        }

        std::cerr << "-> { ";

        for(const auto& i : value) {
            std::cerr << i << " ";
        }

        std::cerr << "}" << std::endl;
    }

    // Default tiling for these dependent IndexSpace s 
    TiledIndexSpace tdepJ(dep_IS_J);
    

    // Dependent IndexSpace s constructed for a and b labels
    IndexSpace dep_IS{{tIS, tdepJ}, dep_relation};

    
    // Default tiling for these dependent IndexSpace s 
    TiledIndexSpace tdep_IS(dep_IS);

    // Construct labels for the operations
    auto [i] = tIS.labels<1>("all");
    auto [k] = tIS.labels<1>("virt");
    auto [j] = tdepJ.labels<1>("all");
    auto [a, b] = tdep_IS.labels<2>("all");


    // Main computation tensors (can be passed as parameters)
#if 0
    Tensor<double> EMP2{i, j(i), a(i,j), b(i,j)};
    Tensor<double> R{i, j(i), a(i,j), b(i,j)};

    auto ec = make_execution_context();
    EMP2.allocate(&ec);
    R.allocate(&ec);

    Scheduler{ec}
        (EMP2() = 1.0)
        (EMP2(k, j(k), a(k,j), b(k,j)) = 42.0)
        (R(i, j(i), a(i,j), b(i,j)) = 2 * EMP2(i, j(i), a(i,j), b(i,j)))
    .execute();

    std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
    print_tensor(EMP2);
    std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
    print_tensor(R);
#else
    Tensor<double> EMP2{i, j(i), a(i,j), b(i,j)};
    Tensor<double> R{i, j(i), a(i,j), b(i,j)};
    Tensor<double> G{i, j(i), a(i,j), b(i,j)};
    Tensor<double> T{i, j(i), a(i,j), b(i,j)};

    // Temporary tensors
    Tensor<double> T_prime{i, j(i), a(i,j), b(i,j)};
    Tensor<double> Temp{i, j(i), a(i,j), b(i,j)}; 
    
    // Construct an ExecutionContext 
    auto ec = make_execution_context();
    // Construct a Scheduler
    Scheduler sch{ec};
    // Assuming these tensors are filled (here we fill them with some values)
    sch.allocate(R, G, T)
        (R() = 42.0)
        (G() = 10.0)
        (T() = 1.0)
    .execute();

    // Main computation for calculating closed-shell PNO-MP2 energy
    // auto EMP2 = (G("i,j")("a,b") + R("i,j")("a,b")).dot(2 * T("i,j")("a,b") - T("i,j")("b,a"));
    sch.allocate(EMP2, T_prime, Temp)
        (T_prime(i, j(i), a(i,j), b(i,j)) = 2.0 * T(i, j(i), a(i,j), b(i,j)))
        (T_prime(i, j(i), a(i,j), b(i,j)) -= T(i, j(i), b(i,j), a(i,j)))
        (Temp(i, j(i), a(i,j), b(i,j)) = G(i, j(i), a(i,j), b(i,j)))
        (Temp(i, j(i), a(i,j), b(i,j)) += R(i, j(i), a(i,j), b(i,j)))
        (EMP2(i, j(i), a(i,j), b(i,j)) += Temp(i, j(i), a(i,j), b(i,j)) * T_prime(i, j(i), a(i,j), b(i,j)))
    .execute();

    std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
    print_tensor(EMP2);
#endif
}
*/

TEST_CASE("GitHub Issues") {

    tamm::ProcGroup pg{GA_MPI_Comm()};
    auto *pMM = tamm::MemoryManagerLocal::create_coll(pg);
    tamm::Distribution_NW dist;
    RuntimeEngine re;
    tamm::ExecutionContext ec(pg, &dist, pMM, &re);

    tamm::TiledIndexSpace X{tamm::IndexSpace{tamm::range(0, 4)}};
    tamm::TiledIndexSpace Y{tamm::IndexSpace{tamm::range(0, 3)}};
    auto [i,j] = X.labels<2>("all");
    auto [a] = Y.labels<1>("all");

    Tensor<double> A{X,X,Y};
    Tensor<double> B{X,X};

    tamm::Scheduler{ec}.allocate(A,B)
    (A() = 3.0)
    (B() = 0.0)
    (B(i,j) = A(i,j,a))
    // (B(i,j) += A(i,j,a))
    .execute();

    // std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
    //print_tensor(B);
}

TEST_CASE("Slack Issues") {
    using tensor_type = Tensor<double>;
    std::cerr << "Slack Issue Start" << std::endl;
    auto ec = make_execution_context();
    Scheduler sch{ec};

    tensor_type initialMO_state;

    IndexSpace AOs_{range(0, 10)};
    IndexSpace MOs_{range(0, 10),
                   {{"O", {range(0, 5)}},
                   {"V", {range(5, 10)}}
    }};

    TiledIndexSpace tAOs{AOs_};
    TiledIndexSpace tMOs{MOs_};
    TiledIndexSpace tXYZ{IndexSpace{range(0,3)}};

    tensor_type D{tXYZ, tAOs, tAOs};
    tensor_type C{tAOs,tMOs};
    tensor_type W{tMOs("O"),tMOs("O")};

    sch.allocate(C, W, D)
        (C() = 42.0)
        (W() = 1.0)
        (D() = 1.0)
    .execute();

    auto xyz = tXYZ;
    auto AOs = C.tiled_index_spaces()[0];
    auto MOs = C.tiled_index_spaces()[1]("O");
    
    initialMO_state = tensor_type{xyz, MOs, MOs};
    tensor_type tmp{xyz, AOs, MOs};

    auto [x] = xyz.labels<1>("all");
    auto [mu, nu] = AOs.labels<2>("all");
    auto [i, j] = MOs.labels<2>("all");     
    
    sch.allocate(initialMO_state, tmp)
        (tmp(x, mu, i) = D(x, mu, nu) * C(nu, i))
        (initialMO_state(x, i, j) = C(mu, i) * tmp(x, mu, j))
    .execute();

    //print_tensor(initialMO_state);

    auto X = initialMO_state.tiled_index_spaces()[0];
    auto n_MOs = W.tiled_index_spaces()[0];
    auto n_LMOs = W.tiled_index_spaces()[1];

    auto [x_] = X.labels<1>("all");
    auto [r_,s_] = n_MOs.labels<2>("all",0);
    auto [i_,j_] = n_LMOs.labels<2>("all",10);

    tensor_type initW{X, n_MOs, n_LMOs};
    tensor_type WinitW{X, n_LMOs, n_LMOs};

    sch.allocate(initW, WinitW)
        (initW(x_,r_,i_) = initialMO_state(x_,r_,s_) * W(s_,i_))
        (WinitW(x_,i_,j_) = W(r_,i_) * initW(x_,r_,j_))
    .execute();
}

TEST_CASE("Slicing examples") {
    IndexSpace AOs{range(0, 10)};
    IndexSpace MOs{range(0, 10),
                   {{"O", {range(0, 5)}},
                   {"V", {range(5, 10)}}
    }};

    TiledIndexSpace tAOs{AOs};
    TiledIndexSpace tMOs{MOs};

    Tensor<double> A{tMOs};
    Tensor<double> B{tMOs, tMOs};

    auto ec = make_execution_context();

    Scheduler sch{ec};

    sch.allocate(A, B)
        (A() = 0.0)
        (B() = 4.0)
    .execute();
    
    auto [i] = tMOs.labels<1>("all");
    auto [j] = tMOs.labels<1>("O");
    auto [k] = tMOs.labels<1>("V");

    sch
        (B(j,j) = 42.0)
        (B(k,k) = 21.0)
        (A(i) = B(i, i))
        // (A() = B(i, i))
        // (B(i,i) = A(i))
    .execute();

    //print_tensor(A);
    //print_tensor(B);
}

TEST_CASE("Fill tensors using lambda functions") {
    IndexSpace AOs{range(0, 10)};
    IndexSpace MOs{range(0, 10),
                   {{"O", {range(0, 5)}},
                   {"V", {range(5, 10)}}
    }};

    TiledIndexSpace tAOs{AOs};
    TiledIndexSpace tMOs{MOs};

    Tensor<double> A{tAOs, tAOs};
    Tensor<double> B{tMOs, tMOs};

    auto ec = make_execution_context();

    A.allocate(&ec);
    B.allocate(&ec);


    update_tensor(A(), lambda_function);
    // std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
    //print_tensor(A);

    Scheduler{ec}
        (A() = 0)
    .execute();
    // std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
    //print_tensor(A);


    update_tensor(A(), l_func<9>);
    // std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
    // print_tensor(A);

    auto i = tAOs.label("all");

    update_tensor(A(i,i), lambda_function);
    // std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
    //print_tensor(A);
}

TEST_CASE("SCF Example Implementation") {

    using tensor_type = Tensor<double>;
    std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
    std::cerr << "SCF Example Implementation" << std::endl;

    IndexSpace AUXs_{range(0, 7)};
    IndexSpace AOs_{range(0, 7)};
    IndexSpace MOs_{range(0, 10),
                   {{"O", {range(0, 5)}},
                    {"V", {range(5, 10)}}
    }};

    TiledIndexSpace Aux{AUXs_};
    TiledIndexSpace AOs{AOs_};
    TiledIndexSpace tMOs{MOs_};

#if 1

    std::map<IndexVector, TiledIndexSpace> dep_nu_mu_q{
        {
            {{0}, TiledIndexSpace{AOs, IndexVector{0,3,4}}},           
            {{2}, TiledIndexSpace{AOs, IndexVector{0,2}}},
            {{3}, TiledIndexSpace{AOs, IndexVector{1,3,5}}},
            {{4}, TiledIndexSpace{AOs, IndexVector{3,5}}},
            {{5}, TiledIndexSpace{AOs, IndexVector{1,2}}},
            {{6}, TiledIndexSpace{AOs, IndexVector{2}}},

        }
    };

    std::map<IndexVector, TiledIndexSpace> dep_nu_mu_d{
        {
            {{0}, TiledIndexSpace{AOs, IndexVector{1,3,5}}},
            {{1}, TiledIndexSpace{AOs, IndexVector{0,1,2}}},
            {{2}, TiledIndexSpace{AOs, IndexVector{0,2,4}}},
            {{3}, TiledIndexSpace{AOs, IndexVector{1,6}}},
            {{4}, TiledIndexSpace{AOs, IndexVector{3,5}}},
            // {{5}, TiledIndexSpace{AOs, IndexVector{0,1,2}}},
            {{6}, TiledIndexSpace{AOs, IndexVector{0,1,2}}}
        }
    };

    std::map<IndexVector, TiledIndexSpace> dep_nu_mu_c{
        {
            {{0}, TiledIndexSpace{AOs, IndexVector{3}}},
            {{2}, TiledIndexSpace{AOs, IndexVector{0,2}}},
            {{3}, TiledIndexSpace{AOs, IndexVector{1}}},
            {{4}, TiledIndexSpace{AOs, IndexVector{3,5}}},
            // {{5}, TiledIndexSpace{AOs, IndexVector{1,2}}},
            {{6}, TiledIndexSpace{AOs, IndexVector{2}}}
        }
    };

    TiledIndexSpace tSubAO_AO_Q{AOs, {AOs}, dep_nu_mu_q};

    TiledIndexSpace tSubAO_AO_D{AOs, {AOs}, dep_nu_mu_d};

    // TiledIndexSpace tSubAO_AO_C{AOs, {AOs}, dep_nu_mu_c};
    auto tSubAO_AO_C = tSubAO_AO_Q.intersect_tis(tSubAO_AO_D);
    // auto tSubAO_AO_C = tSubAO_AO_D.intersect_tis(tSubAO_AO_Q);

    auto X = Aux.label("all",0);
    auto mu = AOs.label("all",1);
    auto nu_for_Q = tSubAO_AO_Q.label("all",0);
    auto nu_for_D = tSubAO_AO_D.label("all",0);
    auto nu_for_C = tSubAO_AO_C.label("all",0);

    tensor_type Q{X, mu, nu_for_Q(mu)};
    tensor_type D{mu, nu_for_D(mu)};
    tensor_type C{X, mu, nu_for_C(mu)};

    auto ec = make_execution_context();
    Scheduler sch{ec};
    
    Q.allocate(&ec);
    D.allocate(&ec);
    C.allocate(&ec);
    
    sch  
    (D() = 42.0)
    (Q() = 2.0)
    (C(X, mu, nu_for_C(mu)) = Q(X, mu, nu_for_C(mu)) * D(mu, nu_for_C(mu)))
    .execute();

    std::cerr << "Tensor C:" << std::endl;
    print_tensor(C);
    std::cerr << "Tensor D" << std::endl;
    print_tensor(D);
    std::cerr << "Tensor Q" << std::endl;
    print_tensor(Q);

#else
    std::map<IndexVector, IndexSpace> dep_mu_i;
    for(const auto& idx : MOs_) {
        if(idx%2 == 0)
            dep_mu_i.insert({{idx}, IndexSpace{AOs_, range(0,3)}});
        else 
            dep_mu_i.insert({{idx}, IndexSpace{AOs_, range(3,7)}});
    }

    std::map<IndexVector, IndexSpace> dep_nu_mu ;
    for(const auto& idx : AOs){
        if(idx < 4)
            dep_nu_mu.insert({{idx}, IndexSpace{AOs_, range(3,7)}});
        else
            dep_nu_mu.insert({{idx}, IndexSpace{AOs_, range(0,3)}});
    }

    IndexSpace subAO_MO{{tMOs}, AOs_, dep_mu_i};
    IndexSpace subAO_AO{{AOs}, AOs_, dep_nu_mu};

    TiledIndexSpace tSubAO_MO{subAO_MO};
    TiledIndexSpace tSubAO_AO{subAO_AO};

    auto [P, Q] = Aux.labels<2>("all",0); 
    auto [mu] = tSubAO_MO.labels<1>("all",2);
    auto [nu] = tSubAO_AO.labels<1>("all",3);
    auto [i] = tMOs.labels<1>("O");

    tensor_type pI{Q, mu, nu(mu)};
    tensor_type C{mu(i), i};

    tensor_type CI{Q, i, nu(i)};
    
    tensor_type Linv{Aux, Aux};

    
    tensor_type D{Aux, tMOs, AOs};
    tensor_type d{Aux};
    tensor_type dL{Aux};
    tensor_type J{AOs, AOs};
    tensor_type K{AOs, AOs};
    
    auto ec = make_execution_context();
    Scheduler sch{ec};

    // Q(X, mu, nu_for_Q(mu)) * D(mu, nu_for_D(mu));
    // Q(X, mu, nu) * D(mu, nu_for_D(mu));

    // sch.allocate(CI, D, d, dL, J, K)
    // (CI(Q, i, nu(i)) = C(mu(i), i) * pI(Q, mu(i), nu(mu))
    // (D(P, i, mu) = Linv(P, Q) * CI(Q, i, mu))
    // (d(P) = D(P, i, mu) * C(mu, i))
    // (dL(Q) = d(P) * Linv(P, Q))
    // (J(mu, nu) = dL(P) * pI(P, mu, nu))
    // (J(mu, nu) = dL(P) * pI(P, mu, nu))
    // (K(mu, nu) = D(P, i, mu) * D(P, i, nu))
    // .execute();

    // std::cerr << __FUNCTION__ << " " << __LINE__ << std::endl;
    // print_tensor(CI);

#endif
}
#endif


void print_dependency(const TiledIndexSpace& tis) {
    auto dep_map = tis.tiled_dep_map();
    std::cerr << "Dependency Map" << std::endl;
   
    for(const auto& [key, subtis] : dep_map) {
        std::cerr << "( ";
        for(const auto& idx : key) {
            std::cerr << idx << " ";
        }
        std::cerr << ") -> ";

        std::cerr << "{ ";
        for(const auto& idx : subtis.ref_indices()) {
            std::cerr << idx << " ";
        }
        std::cerr << "}" << std::endl;
    }
}

TEST_CASE("TiledIndexSpace operations test") {

    TiledIndexSpace AOs{IndexSpace{range(7)}};
    TiledIndexSpace MOs{IndexSpace{range(4)}};

    std::map<IndexVector, TiledIndexSpace> new_dep{
        {
            {{0, 0}, TiledIndexSpace{AOs, IndexVector{0,3,4}}},           
            {{1, 5}, TiledIndexSpace{AOs, IndexVector{0,3,6}}},
            {{2, 0}, TiledIndexSpace{AOs, IndexVector{1,3,5}}},
            {{2, 5}, TiledIndexSpace{AOs, IndexVector{0,5}}}
        }
    };


    std::map<IndexVector, TiledIndexSpace> dep_nu_mu_q{
        {
            {{0}, TiledIndexSpace{AOs, IndexVector{0,3,4}}},           
            {{1}, TiledIndexSpace{AOs, IndexVector{0,3,6}}},
            {{2}, TiledIndexSpace{AOs, IndexVector{1,3,5}}},
            // {{4}, TiledIndexSpace{AOs, IndexVector{3,5}}},
            // {{5}, TiledIndexSpace{AOs, IndexVector{1,2}}},
            // {{6}, TiledIndexSpace{AOs, IndexVector{2}}},

        }
    };

    std::map<IndexVector, TiledIndexSpace> dep_nu_mu_d{
        {
            {{0}, TiledIndexSpace{AOs, IndexVector{1,3,5}}},
            {{1}, TiledIndexSpace{AOs, IndexVector{0,1,2}}},
            {{2}, TiledIndexSpace{AOs, IndexVector{0,2,4}}},
            {{3}, TiledIndexSpace{AOs, IndexVector{1,6}}},
            {{4}, TiledIndexSpace{AOs, IndexVector{3,5}}},
            // {{5}, TiledIndexSpace{AOs, IndexVector{0,1,2}}},
            {{6}, TiledIndexSpace{AOs, IndexVector{0,1,2}}}
        }
    };

    std::map<IndexVector, TiledIndexSpace> dep_nu_mu_c{
        {
            {{0}, TiledIndexSpace{AOs, IndexVector{3}}},
            {{2}, TiledIndexSpace{AOs, IndexVector{0,2}}},
            {{3}, TiledIndexSpace{AOs, IndexVector{1}}},
            {{4}, TiledIndexSpace{AOs, IndexVector{3,5}}},
            // {{5}, TiledIndexSpace{AOs, IndexVector{1,2}}},
            {{6}, TiledIndexSpace{AOs, IndexVector{2}}}
        }
    };

    TiledIndexSpace test_tis{AOs, {MOs, AOs}, new_dep};

    TiledIndexSpace tSubAO_AO_Q{AOs, {MOs}, dep_nu_mu_q};

    TiledIndexSpace tSubAO_AO_D{AOs, {AOs}, dep_nu_mu_d};

    TiledIndexSpace tSubAO_AO_C{AOs, {AOs}, dep_nu_mu_c};

    // auto intersect = tSubAO_AO_Q.intersect_tis(tSubAO_AO_D);
    // REQUIRE(intersect == tSubAO_AO_C);
    std::cerr << "tSubAO_AO_Q ";
    print_dependency(tSubAO_AO_Q);
    
    auto inv_tSubAO_AO_Q = invert_tis(tSubAO_AO_Q);
    std::cerr << "inv_tSubAO_AO_Q ";
    print_dependency(inv_tSubAO_AO_Q);

    auto comp_tSubAO_AO_Q_D = compose_tis(tSubAO_AO_Q, tSubAO_AO_D);
    std::cerr << "tSubAO_AO_D ";
    print_dependency(tSubAO_AO_D);
    std::cerr << "comp_tSubAO_AO_Q_D ";
    print_dependency(comp_tSubAO_AO_Q_D);

    auto union_tSubAO_AO_D_C = union_tis(tSubAO_AO_D, tSubAO_AO_C);
    std::cerr << "tSubAO_AO_D ";
    print_dependency(tSubAO_AO_D);
    std::cerr << "tSubAO_AO_C ";
    print_dependency(tSubAO_AO_C);
    std::cerr << "union_tSubAO_AO_D_C ";
    print_dependency(union_tSubAO_AO_D_C);

    auto project_test_tis = project_tis(test_tis, MOs);
    std::cerr << "test_tis ";
    print_dependency(test_tis);
    std::cerr << "project_test_tis ";
    print_dependency(project_test_tis);

    auto project_MO_Q = project_tis(tSubAO_AO_Q, MOs);
    std::cerr << "tSubAO_AO_Q ";
    print_dependency(tSubAO_AO_Q);
    std::cerr << "project_MO_Q " << std::endl;
    std::cerr << "{ ";
    for(const auto& idx : project_MO_Q.ref_indices()) {
        std::cerr << idx << " ";
    }
    std::cerr << "}" << std::endl;

    TiledIndexSpace tis_1{AOs, IndexVector{1,2,5}};
    TiledIndexSpace tis_2{AOs, IndexVector{2,3,6}};

    auto u_tis12 = union_tis(tis_1, tis_2);
    std::cerr << "u_tis12 " << std::endl;
    std::cerr << "{ ";
    for(const auto& idx : u_tis12.ref_indices()) {
        std::cerr << idx << " ";
    }
    std::cerr << "}" << std::endl;

    TiledIndexLabel A = tSubAO_AO_Q.label();
    for(Index i : MOs) {
        std::cerr << "{ ";
        for(const auto& idx : A(i).tiled_index_space().ref_indices()) {
            std::cerr << idx << " ";
        }
        std::cerr << "}" << std::endl;
        // Tensor<double> t1{};
    }
}

#if 0
using DepMap= std::map<IndexVector, TiledIndexSpace>;

DepMap LMO_domain(){
    DepMap res;
    // DepMap computation
    return res;
}

DepMap AO_domain(){
    DepMap res;
    // DepMap computation
    return res;
}

DepMap fitting_domain(){
    DepMap res;
    // DepMap computation
    return res;
}

Tensor<T> cholesky(const Tensor<T>& tens){
    Tensor<T> res;

    return res;
}

TEST_CASE("Sample code for Local HF") {
    // TAMM Scheduler construction
    auto ec = tamm::make_execution_context();
    Scheduler sch{ec};

    // Dummy TiledIndexSpaces
    TiledIndexSpace TAO{IndexSpace{range(10)}};
    TiledIndexSpace TMO{IndexSpace{range(10)}};

    // Local SCF TAMM Pseudo-code
    
    // Input dense C tensor
    Tensor<T> LMO{TAO, TMO};  //dense

    //LMO_domain(): chooses AOs i -> mu 
    auto lmo_dep_map = LMO_domain();

    // TiledIndexSpace lmo_domain{mu(i)}; //construct using explicit loop
    TiledIndexSpace lmo_domain{TAO, {TMO}, lmo_dep_map}; //construct using explicit loop
    
    //LMO_renormalize() {
        auto [i] = TMO.labels<1>("all");
        auto [mu, nu] = lmo_domain.labels<2>("all");
        auto [mu_p] = TAO.labels<1>("all");

        Tensor<T> S_A{i, mu(i), mu(i)};
        Tensor<T> S_v{i, mu_p, mu(i)};
        Tensor<T> C{i, mu_p};   //column of LMO

        //solved using Eigen

        // Sparsified LMO 
        Tensor<T> LMO_renorm{mu(i), i}; //sparsified LMO
        
        sch
        .allocate(LMO_renorm)
            (LMO_renorm(mu(i), i) = LMO(mu(i), i))
        .execute();
    // }


    //AO_domain(): constructs ao->ao index space
    auto ao_screen_dep_map = AO_domain();

    // TiledIndexSpace ao_int_screening{nu(mu)}; //ao->ao
    TiledIndexSpace ao_int_screening{TAO, {TAO}, ao_screen_dep_map};

    // //chain_maps(): compose lmo->ao and ao->ao
    auto [nu_p] = ao_int_screening.labels<1>("all");

    // TiledIndexSpace ao_domain{nu(i)}; //mo->ao
    // compose using labels
    auto ao_domain = compose_lbl(mu(i), nu_p(mu)); // nu(i) -> return label 
    // compose using TiledIndexSpaces
    // auto ao_domain = compose_tis(lmo_domain, ao_int_screening); // -> return tis

    //fitting domain
    // IndexSpace fb; //fitting basis. this is already available and used as input

    auto lmo_to_fit_dep_map = fitting_domain();

    // Output:
    // TiledIndexSpace lmo_to_fit{A(i)}; // mo-> fitting basis
    TiledIndexSpace lmo_to_fit{TAO, {TMO}, lmo_to_fit_dep_map}; //mo->fitting basis

    //continuing with build_K. first contraction “transformation step”

    // TiledIndexSpace ao_to_lmo{i(mu)}; // 
    // invert using labels
    auto ao_to_lmo= invert_lbl(mu(i)); // i(mu)
    // invert using TiledIndexSpaces
    // auto ao_to_lmo= invert_tis(lmo_domain);

    // IndexLabel i(mu);//ao_to_lmo
    auto [A, B] = lmo_to_fit.labels<2>("all");


    //Construct matrix of Coulomb metric, J, only compute for AB pairs which share an lmo
    auto fit_to_lmo = invert_lbl(A(i));               // i(A)
   
    auto fit_to_ao  = compose_lbl(fit_to_lmo, mu(i)); // mu(A)
    auto B_p = compose_lbl(fit_to_lmo, A(i));  // B(A)

    // auto [B_p] = fit_to_fit.labels<1>("all");

    // Input X (tensor with lambda function that calls libint)
    Tensor<T> X{A(i), mu(i), nu(i)}; // internally project on i ?
    // input J
    Tensor<T> J{A, B_p(A)};

    // results
    Tensor<T> Q{A(i), mu(i), i};
    Tensor<T> QB{B(i), mu(i), i};
    Tensor<T> K{mu(i), nu(i)};

    sch.allocate(Q, QB, K);
    // foreach Index i in TMO:
    for(Index i_val : TMO){
        Tensor<T> J_i{A(i_val), B(i_val)};
        Tensor<T> G_i_inv{A(i_val), B(i_val)};
        sch
        .allocate(J_i, G_i_inv)         // Q: how to allocate within a loop?
            (Q(A(i_val), mu(i_val), i_val) = X(A(i_val), mu(i_val), nu(i_val)) * C(nu(i_val), i_val))
            (J_i(A(i_val), B(i_val)) = J(A(i_val), B(i_val))) 
        .execute();

        G_i_inv = invert_tensor(cholesky(J_i));

        sch
            (QB(B(i_val), mu(i_val), i_val) += G_i_inv(B(i_val), A(i_val)) * Q(A(i_val), mu(i_val), i_val))
            // (K(mu, nu(mu)) += QB(A(i), mu(i), i) * QB(A(i), nu(i), i)) //nu(mu) is a dependent representation of the sparsity
            (K(mu(i_val), nu(i_val)) += QB(A(i_val), mu(i_val), i_val) * QB(A(i_val), nu(i_val), i_val))
        .deallocate(J_i, G_i_inv)
        .execute();
    }

}
#endif

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