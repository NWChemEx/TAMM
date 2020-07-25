#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"
#include "ga-mpi.h"
#include "ga.h"
#include "macdecls.h"
#include "mpi.h"
#include "tamm/tamm.hpp"

#include <complex>
#include <string>

/**
 * @brief Tests for operations
 *
 * @todo Test operations on subspaces
 *
 * @todo Test for different tile sizes
 *
 * @todo Test various permutation orders
 *
 */

using namespace tamm;

using complex_single = std::complex<float>;
using complex_double = std::complex<double>;


template<typename T>
void print_tensor(Tensor<T>& t) {
    for(auto it : t.loop_nest()) {
        TAMM_SIZE size = t.block_size(it);
        std::vector<T> buf(size);
        t.get(it, buf);
        std::cout << "block" << it;
        for(TAMM_SIZE i = 0; i < size; i++) std::cout << buf[i] << std::endl;
    }
}

template<typename T>
void check_value(LabeledTensor<T> lt, T val) {
    LabelLoopNest loop_nest{lt.labels()};
    for(const auto& itval : loop_nest) {
        const IndexVector blockid = internal::translate_blockid(itval, lt);
        size_t size               = lt.tensor().block_size(blockid);
        std::vector<T> buf(size);
        lt.tensor().get(blockid, buf);
        for(TAMM_SIZE i = 0; i < size; i++) {
            if constexpr(tamm::internal::is_complex_v<T>) {
                REQUIRE(std::fabs(buf[i].real() - val.real()) < 1.0e-10);                
            } else {
                REQUIRE(std::fabs(buf[i] - val) < 1.0e-10);
            }
        }
    }
}

template<typename T>
void check_value(Tensor<T>& t, T val) {
    check_value(t(), val);
}

template<typename T>
void test_ops(const TiledIndexSpace& MO) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    const TiledIndexSpace& N = MO("all");

    Tensor<T> d_t1{V, O};
    Tensor<T> d_t2{V, V, O, O};

    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};
    Tensor<T> T1{N, N, N};
    Tensor<T>::allocate(ec, T1);

    //@todo cleanup this file, seperate get,put,setop tests
    T cx              = 42;
    const size_t size = 1000;
    T* buf            = new T[size];
    for(size_t i = 0; i < size; i++) buf[i] = cx++;

    T1.put(IndexVector{1, 0, 1}, buf);

    T* gbuf = new T[size];
    T1.get(IndexVector{1, 0, 1}, {gbuf, size});
    for(size_t i = 0; i < T1.block_size({1, 0, 1}); i++)
        EXPECTS(gbuf[i] == buf[i]);
    Tensor<T>::deallocate(T1);
    Tensor<T> xt1{N, N};
    Tensor<T> xt2{N, N};
    Tensor<T> xt3{N, N};
    // Tensor<T>::allocate(ec,xt1,xt2,xt3);

#if 1
    Scheduler{*ec}
      .allocate(xt1, xt2, xt3)(xt1("n1", "n2") = 2.2)(xt2("n1", "n2") =
                                                        2.0 * xt1("n1", "n2"))
      //(xt3("n1","n2") = 2.0*xt1("n1","nk")*xt2("nk","n2")) //no-op
      //.deallocate(xt3)
      .execute();

    check_value(xt1, 2.2);
    check_value(xt2, 4.4);
#endif

    Tensor<T>::deallocate(xt1, xt2, xt3);
}

int main(int argc, char* argv[]) {

    tamm::initialize(argc, argv);

    doctest::Context context(argc, argv);

    int res = context.run();

    tamm::finalize();

    return res;
}

// TEST_CASE("Test Ops") {
//     // Construction of tiled index space MO from sketch
//     IndexSpace MO_IS{range(0, 10),
//                      {{"occ", {range(0, 5)}}, {"virt", {range(5, 10)}}}};
//     TiledIndexSpace MO{MO_IS, 1};

//     CHECK_NOTHROW(test_ops<double>(MO));
// }

TEST_CASE("Tensor Allocation and Deallocation") {
    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};

    {
        Tensor<double> tensor{};
        Tensor<double>::allocate(ec, tensor);
    }

    ec->flush_and_sync();

    //MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}

#if 1
TEST_CASE("Zero-dimensional ops") {
    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};
    using T              = double;

    IndexSpace IS{range(0, 10)};
    TiledIndexSpace TIS{IS, 1};

    {
        Tensor<T> T1{};
        Tensor<T>::allocate(ec, T1);
        Scheduler{*ec}(T1() = 42).execute();
        check_value(T1, (T)42.0);
        Tensor<T>::deallocate(T1);
    }

    {
        Tensor<T> T1{}, T2{}, T3{};
        Tensor<T>::allocate(ec, T1, T2, T3);
        auto lambda1 = [](Tensor<T>& t, const IndexVector& iv,
                          std::vector<T>& buf) {
            for(auto& v : buf) v = 0;
        };
        auto lambda2 = [](const Tensor<T>& t, const IndexVector& lhs_iv,
                          std::vector<T>& lhs_buf, const IndexVector rhs_iv[],
                          std::vector<T> rhs_buf[]) {
            std::copy(rhs_buf[0].begin(), rhs_buf[0].end(), lhs_buf.begin());
        };
        auto lambda3 = [](const Tensor<T>& t, const IndexVector& lhs_iv,
                          std::vector<T>& lhs_buf, const IndexVector rhs_iv[],
                          std::vector<T> rhs_buf[]) {
            for(size_t i = 0; i < lhs_buf.size(); ++i) {
                lhs_buf[i] = rhs_buf[0][i] + rhs_buf[1][i];
            }
        };

        Scheduler{*ec}(T1() = 42)(T2() = 43)(T3() = 44).execute();
        // ScanOp
        Scheduler{*ec}.gop(T1(), lambda1).execute();
        check_value(T1, (T)42);
        // MapOp
        Scheduler{*ec}
          .gop(T1(), std::array<decltype(T2()), 1>{T2()}, lambda2)
          .execute();
        check_value(T1, (T)43);
        Scheduler{*ec}
          .gop(T1(), std::array<decltype(T2()), 2>{T2(), T3()}, lambda3)
          .execute();
        check_value(T1, (T)43 + 44);
        Tensor<T>::deallocate(T1, T2, T3);
    }

    {
        Tensor<T> T1{}, T2{};
        Scheduler{*ec}
          .allocate(T1, T2)(T2() = 42)(T1() = T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)42.0);
        Tensor<T>::deallocate(T1);
    }

    {
        Tensor<T> T1{}, T2{};
        Scheduler{*ec}
          .allocate(T1, T2)(T1() = 3)(T2() = 42)(T1() += T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)45.0);
        Tensor<T>::deallocate(T1);
    }

    {
        Tensor<T> T1{}, T2{};
        Scheduler{*ec}
          .allocate(T1, T2)(T1() = 42)(T2() = 3)(T1() += 2.5 * T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)49.5);
        Tensor<T>::deallocate(T1);
    }

    {
        Tensor<T> T1{}, T2{};
        Scheduler{*ec}
          .allocate(T1, T2)(T1() = 42)(T2() = 3)(T1() -= T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)39.0);
        Tensor<T>::deallocate(T1);
    }

    {
        Tensor<T> T1{}, T2{};
        Scheduler{*ec}
          .allocate(T1, T2)(T1() = 42)(T2() = 3)(T1() -= 4 * T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)30.0);
        Tensor<T>::deallocate(T1);
    }

    {
        Tensor<T> T1{}, T2{}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 3)(T3() = 5)(T1() += T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)15.0);
        Tensor<T>::deallocate(T1);
    }

    //MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}
#endif

TEST_CASE("Zero-dimensional ops with flush and sync deallocation") {
    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};
    using T              = double;

    IndexSpace IS{range(0, 10)};
    TiledIndexSpace TIS{IS, 1};

    {
        Tensor<T> T1{};
        Tensor<T>::allocate(ec, T1);
        Scheduler{*ec}(T1() = 42).execute();
        check_value(T1, (T)42.0);
    }

    ec->flush_and_sync();

    {
        Tensor<T> T1{}, T2{}, T3{};
        Tensor<T>::allocate(ec, T1, T2, T3);
        auto lambda1 = [](Tensor<T>& t, const IndexVector& iv,
                          std::vector<T>& buf) {
            for(auto& v : buf) v = 0;
        };
        auto lambda2 = [](const Tensor<T>& t, const IndexVector& lhs_iv,
                          std::vector<T>& lhs_buf, const IndexVector rhs_iv[],
                          std::vector<T> rhs_buf[]) {
            std::copy(rhs_buf[0].begin(), rhs_buf[0].end(), lhs_buf.begin());
        };
        auto lambda3 = [](const Tensor<T>& t, const IndexVector& lhs_iv,
                          std::vector<T>& lhs_buf, const IndexVector rhs_iv[],
                          std::vector<T> rhs_buf[]) {
            for(size_t i = 0; i < lhs_buf.size(); ++i) {
                lhs_buf[i] = rhs_buf[0][i] + rhs_buf[1][i];
            }
        };

        Scheduler{*ec}(T1() = 42)(T2() = 43)(T3() = 44).execute();
        // ScanOp
        Scheduler{*ec}.gop(T1(), lambda1).execute();
        check_value(T1, (T)42);
        // MapOp
        Scheduler{*ec}
          .gop(T1(), std::array<decltype(T2()), 1>{T2()}, lambda2)
          .execute();
        check_value(T1, (T)43);
        Scheduler{*ec}
          .gop(T1(), std::array<decltype(T2()), 2>{T2(), T3()}, lambda3)
          .execute();
        check_value(T1, (T)43 + 44);
    }

    ec->flush_and_sync();

    {
        Tensor<T> T1{}, T2{};
        Scheduler{*ec}
          .allocate(T1, T2)(T2() = 42)(T1() = T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)42.0);
    }

    ec->flush_and_sync();

    {
        Tensor<T> T1{}, T2{};
        Scheduler{*ec}
          .allocate(T1, T2)(T1() = 3)(T2() = 42)(T1() += T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)45.0);
        Tensor<T>::deallocate(T1);
    }

    // test flush and sync with tensor that has been deallocated (T1 above)

    {
        Tensor<T> T1{}, T2{};
        Scheduler{*ec}
          .allocate(T1, T2)(T1() = 42)(T2() = 3)(T1() += 2.5 * T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)49.5);
    }
    
    ec->flush_and_sync();

    {
        Tensor<T> T1{}, T2{};
        Scheduler{*ec}
          .allocate(T1, T2)(T1() = 42)(T2() = 3)(T1() -= T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)39.0);
    }

    {
        Tensor<T> T1{}, T2{};
        Scheduler{*ec}
          .allocate(T1, T2)(T1() = 42)(T2() = 3)(T1() -= 4 * T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)30.0);
    }

    {
        Tensor<T> T1{}, T2{}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 3)(T3() = 5)(T1() += T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)15.0);
    }

    ec->flush_and_sync();

    //MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}

template<typename T>
bool test_setop(ExecutionContext* ec, Tensor<T> T1, LabeledTensor<T> LT1,
                const std::vector<LabeledTensor<T>>& rest_lts = {}) {
    bool success = true;
    try {
        Tensor<T>::allocate(ec, T1);
        try {
            Scheduler{*ec}(T1() = -1.0)(LT1 = 42).execute();
            check_value(LT1, (T)42.0);
            for(auto lt : rest_lts) { check_value(lt, (T)-1.0); }
        } catch(std::string& e) {
            std::cerr << "Caught exception: " << e << "\n";
            success = false;
        }
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        success = false;
    }
    return success;
}

template<typename T>
bool test_mapop(ExecutionContext* ec, Tensor<T> T1, LabeledTensor<T> LT1,
                Tensor<T> T2, LabeledTensor<T> LT2,
                const std::vector<LabeledTensor<T>>& rest_lts = {}) {
    bool success = true;
    try {
        Tensor<T>::allocate(ec, T1);
        Tensor<T>::allocate(ec, T2);
        try {
            auto lambda = [](const Tensor<T>& t, const IndexVector& lhs_iv,
                             std::vector<T>& lhs_buf,
                             const IndexVector rhs_iv[],
                             std::vector<T> rhs_buf[]) {
                std::copy(rhs_buf[0].begin(), rhs_buf[0].end(),
                          lhs_buf.begin());
            };
            Scheduler{*ec}(T1() = -1.0)(T2() = 1.0)
              .gop(LT1, std::array<decltype(LT2), 1>{LT2}, lambda)
              .execute();
            check_value(LT1, (T)1);
            for(auto lt : rest_lts) { check_value(lt, (T)-1.0); }
        } catch(std::string& e) {
            std::cerr << "Caught exception: " << e << "\n";
            success = false;
        }
        Tensor<T>::deallocate(T1);
        Tensor<T>::deallocate(T2);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        success = false;
    }
    return success;
}

template<typename T>
bool test_addop(ExecutionContext* ec, Tensor<T> T1, Tensor<T> T2,
                LabeledTensor<T> LT1, LabeledTensor<T> LT2,
                std::vector<LabeledTensor<T>> rest_lts = {}) {
    bool success = true;
    Tensor<T>::allocate(ec, T1, T2);

    try {
        Scheduler{*ec}(T1() = -1.0)(LT2 = 42)(LT1 = LT2).execute();
        check_value(LT1, (T)42.0);
        for(const auto& lt : rest_lts) { check_value(lt, (T)-1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 0. Exception: " << e << "\n";
        success = false;
    }

    try {
        success = true;
        Scheduler{*ec}(T1() = -1.0)(LT1 = 4)(LT2 = 42)(LT1 += LT2).execute();
        check_value(LT1, (T)46.0);
        for(const auto& lt : rest_lts) { check_value(lt, (T)-1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 1. Exception: " << e << "\n";
        success = false;
    }

    try {
        success = true;
        Scheduler{*ec}(T1() = -1.0)(LT1 = 4)(LT2 = 42)(LT1 += 3 * LT2).execute();
        check_value(T1, (T)130.0);
        for(const auto& lt : rest_lts) { check_value(lt, (T)-1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 2. Exception: " << e << "\n";
        success = false;
    }

    try {
        success = true;
        Scheduler{*ec}(T1() = -1.0)(T1() = 4)(T2() = 42)(T1() -= T2()).execute();
        check_value(T1, (T)-38.0);
        for(const auto& lt : rest_lts) { check_value(lt, (T)-1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 3. Exception: " << e << "\n";
        success = false;
    }

    try {
        success = true;
        Scheduler{*ec}(T1() = -1.0)(T1() = 4)(T2() = 42)(T1() += -3.1 * T2())
          .execute();
        check_value(T1, (T)-126.2);
        for(const auto& lt : rest_lts) { check_value(lt, (T)-1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 4. Exception: " << e << "\n";
        success = false;
    }

    try {
        success = true;
        Scheduler{*ec}(T1() = -1.0)(T1() = 4)(T2() = 42)(T1() -= 3.1 * T2())
          .execute();
        check_value(T1, (T)-126.2);
        for(const auto& lt : rest_lts) { check_value(lt, (T)-1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 4. Exception: " << e << "\n";
        success = false;
    }

    Tensor<T>::deallocate(T1, T2);
    return success;
}

// setop with T (call with tilesize 1 and 3)
template<typename T>
void test_setop_with_T(unsigned tilesize) {
    // 0-4 dimensional setops
    // 0-4 dimensional setops

    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};

    IndexSpace IS{range(0, 10),
                  {{"nr1", {range(0, 5)}}, {"nr2", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, tilesize};
    TiledIndexLabel l1, l2, lall, lnone;
    std::tie(l1)   = TIS.labels<1>("nr1");
    std::tie(l2)   = TIS.labels<1>("nr2");
    std::tie(lall) = TIS.labels<1>("all");
    //@todo is there a "none" slice?
    // std::tie(lnone) = TIS.labels<1>("none");

    // 0-dimensional test
    {
        Tensor<T> T1{};
        REQUIRE(test_setop(ec, T1, T1()));
    }

    // 1-dimensional tests
    {
        Tensor<T> T1{TIS};
        REQUIRE(test_setop(ec, T1, T1()));
    }

    {
        Tensor<T> T1{TIS};
        REQUIRE(test_setop(ec, T1, T1(l1)));
    }

    {
        Tensor<T> T1{TIS};
        REQUIRE(test_setop(ec, T1, T1(l1), {T1(l2)}));
    }

    // 2-dimensional tests
    {
        Tensor<T> T1{TIS, TIS};
        REQUIRE(test_setop(ec, T1, T1()));
    }
    {
        Tensor<T> T1{TIS, TIS};
        REQUIRE(test_setop(ec, T1, T1(l1, l1)));
    }

    {
        Tensor<T> T1{TIS, TIS};
        REQUIRE(
          test_setop(ec, T1, T1(l1, l1), {T1(l1, l2), T1(l2, l1), T1(l2, l2)}));
    }
    // 3-dimensional tests
    {
        Tensor<T> T1{TIS, TIS, TIS};
        REQUIRE(test_setop(ec, T1, T1()));
    }
    {
        Tensor<T> T1{TIS, TIS, TIS};
        REQUIRE(test_setop(ec, T1, T1(l1, l2, l2)));
    }

    {
        Tensor<T> T1{TIS, TIS, TIS};
        REQUIRE(test_setop(ec, T1, T1(l1, l2, l2),
                           {T1(l1, l1, l1), T1(l1, l1, l2), T1(l1, l2, l1),
                            T1(l2, l1, l1), T1(l2, l1, l2), T1(l2, l2, l1),
                            T1(l2, l2, l2)}));
    }
    // 4-dimensional tests
    {
        Tensor<T> T1{TIS, TIS, TIS, TIS};
        REQUIRE(test_setop(ec, T1, T1()));
    }
    {
        Tensor<T> T1{TIS, TIS, TIS, TIS};
        REQUIRE(test_setop(ec, T1, T1(l1, l2, l2, l1)));
    }

    {
        Tensor<T> T1{TIS, TIS, TIS, TIS};
        REQUIRE(test_setop(
          ec, T1, T1(l1, l2, l2, l1),
          {T1(l1, l1, l1, l1), T1(l1, l1, l1, l2), T1(l1, l1, l2, l1),
           T1(l1, l1, l2, l2), T1(l1, l2, l1, l1), T1(l1, l2, l1, l2),
           T1(l1, l2, l2, l2), T1(l2, l1, l1, l1), T1(l2, l1, l1, l2),
           T1(l2, l1, l2, l1), T1(l2, l2, l1, l1), T1(l2, l2, l1, l2),
           T1(l2, l2, l2, l1), T1(l2, l2, l2, l2)}));
    }

    //MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}

// mapop with T (call with tilesize 1 and 3)
template<typename T>
void test_mapop_with_T(unsigned tilesize) {
    // 0-4 dimensional setops
    // 0-4 dimensional setops

    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};

    IndexSpace IS{range(0, 10),
                  {{"nr1", {range(0, 5)}}, {"nr2", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, tilesize};
    TiledIndexLabel l1, l2, lall, lnone;
    std::tie(l1)   = TIS.labels<1>("nr1");
    std::tie(l2)   = TIS.labels<1>("nr2");
    std::tie(lall) = TIS.labels<1>("all");
    //@todo is there a "none" slice?
    // std::tie(lnone) = TIS.labels<1>("none");

    // 0-dimensional test
    {
        Tensor<T> T1{}, T2{};
        REQUIRE(test_mapop(ec, T1, T1(), T2, T2()));
    }

    // 1-dimensional tests
    {
        Tensor<T> T1{TIS}, T2{TIS};
        REQUIRE(test_mapop(ec, T1, T1(), T2, T2()));
    }

    {
        Tensor<T> T1{TIS}, T2{TIS};
        REQUIRE(test_mapop(ec, T1, T1(l1), T2, T2(l1)));
    }

    {
        Tensor<T> T1{TIS}, T2{TIS};
        REQUIRE(test_mapop(ec, T1, T1(l1), T2, T2(l1), {T1(l2)}));
    }

    // 2-dimensional tests
    {
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS};
        REQUIRE(test_mapop(ec, T1, T1(), T2, T2()));
    }
    {
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS};
        REQUIRE(test_mapop(ec, T1, T1(l1, l1), T2, T2(l1, l1)));
    }

    {
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS};
        REQUIRE(test_mapop(ec, T1, T1(l1, l1), T2, T2(l1, l1),
                           {T1(l1, l2), T1(l2, l1), T1(l2, l2)}));
    }
    // 3-dimensional tests
    {
        Tensor<T> T1{TIS, TIS, TIS}, T2{TIS, TIS, TIS};
        REQUIRE(test_mapop(ec, T1, T1(), T2, T2()));
    }
    {
        Tensor<T> T1{TIS, TIS, TIS}, T2{TIS, TIS, TIS};
        REQUIRE(test_mapop(ec, T1, T1(l1, l2, l2), T2, T2(l1, l2, l2)));
    }

    {
        Tensor<T> T1{TIS, TIS, TIS}, T2{TIS, TIS, TIS};
        REQUIRE(test_mapop(ec, T1, T1(l1, l2, l2), T2, T2(l1, l2, l2),
                           {T1(l1, l1, l1), T1(l1, l1, l2), T1(l1, l2, l1),
                            T1(l2, l1, l1), T1(l2, l1, l2), T1(l2, l2, l1),
                            T1(l2, l2, l2)}));
    }
    // 4-dimensional tests
    {
        Tensor<T> T1{TIS, TIS, TIS, TIS}, T2{TIS, TIS, TIS, TIS};
        REQUIRE(test_mapop(ec, T1, T1(), T2, T2()));
    }
    {
        Tensor<T> T1{TIS, TIS, TIS, TIS}, T2{TIS, TIS, TIS, TIS};
        REQUIRE(test_mapop(ec, T1, T1(l1, l2, l2, l1), T2, T2(l1, l2, l2, l1)));
    }

    {
        Tensor<T> T1{TIS, TIS, TIS, TIS}, T2{TIS, TIS, TIS, TIS};
        REQUIRE(test_mapop(
          ec, T1, T1(l1, l2, l2, l1), T2, T2(l1, l2, l2, l1),
          {T1(l1, l1, l1, l1), T1(l1, l1, l1, l2), T1(l1, l1, l2, l1),
           T1(l1, l1, l2, l2), T1(l1, l2, l1, l1), T1(l1, l2, l1, l2),
           T1(l1, l2, l2, l2), T1(l2, l1, l1, l1), T1(l2, l1, l1, l2),
           T1(l2, l1, l2, l1), T1(l2, l2, l1, l1), T1(l2, l2, l1, l2),
           T1(l2, l2, l2, l1), T1(l2, l2, l2, l2)}));
    }

    //MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}

// addop with T  (call with tilesize 1 and 3)
template<typename T>
void test_addop_with_T(unsigned tilesize) {
    // 0-4 dimensional addops
    bool failed;
    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};

    IndexSpace IS{range(0, 10),
                  {{"nr1", {range(0, 5)}}, {"nr2", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, tilesize};
    // 0-dimensional test
    {
        Tensor<T> T1{}, T2{};
        test_addop(ec, T1, T2, T1(), T2());
    }

    try {
        failed = false;
        Tensor<T> T1{}, T2{}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() += T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{}, T2{}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() -= T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)-32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{}, T2{}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() += 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(9 + 1.5 * 8 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{}, T2{}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() -= 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(9 - 1.5 * 8 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{}, T2{}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() += 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value(T3, (T)(4 + 1.5 * 9 * 8));
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{}, T2{}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() -= 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value(T3, (T)(4 - 1.5 * 9 * 8));
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    // 1-dimension tests
    {
        Tensor<T> T1{TIS}, T2{TIS};
        test_addop(ec, T1, T2, T1(), T2());
    }

    try {
        failed = false;
        Tensor<T> T1{TIS}, T2{TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() += T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS}, T2{TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() -= T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)-32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS}, T2{TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() += 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(9 + 1.5 * 8 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS}, T2{TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() -= 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(9 - 1.5 * 8 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS}, T2{TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() += 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value(T3, (T)(4 + 1.5 * 10 * 9 * 8));
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS}, T2{TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() -= 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value(T3, (T)(4 - 1.5 * 10 * 9 * 8));
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    // 2-dimension tests
    {
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS};
        test_addop(ec, T1, T2, T1(), T2());
    }

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() += T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() -= T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)-32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() += 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(9 + 1.5 * 8 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() -= 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(9 - 1.5 * 8 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() += 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value(T3, (T)(4 + 1.5 * 100 * 9 * 8));
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() -= 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value(T3, (T)(4 - 1.5 * 100 * 9 * 8));
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    // 3-dimension tests
    {
        Tensor<T> T1{TIS, TIS, TIS}, T2{TIS, TIS, TIS};
        test_addop(ec, T1, T2, T1(), T2());
    }

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS}, T2{TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() += T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS}, T2{TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() -= T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)-32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS}, T2{TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() += 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(9 + 1.5 * 8 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS}, T2{TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() -= 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(9 - 1.5 * 8 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS}, T2{TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() += 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value(T3, (T)(4 + 1.5 * 1000 * 9 * 8));
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS}, T2{TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() -= 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value(T3, (T)(4 - 1.5 * 1000 * 9 * 8));
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    // 4-dimension tests
    {
        Tensor<T> T1{TIS, TIS, TIS, TIS}, T2{TIS, TIS, TIS, TIS};
        test_addop(ec, T1, T2, T1(), T2());
    }

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS, TIS}, T2{TIS, TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() += T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS, TIS}, T2{TIS, TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() -= T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)-32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS, TIS}, T2{TIS, TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() += 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(9 + 1.5 * 8 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS, TIS}, T2{TIS, TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() -= 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(9 - 1.5 * 8 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS, TIS}, T2{TIS, TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() += 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value(T3, (T)(4 + 1.5 * 10000 * 9 * 8));
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS, TIS}, T2{TIS, TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() -= 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value(T3, (T)(4 - 1.5 * 10000 * 9 * 8));
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    //MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}

TEST_CASE("setop with double") {
    test_setop_with_T<double>(1);
    test_setop_with_T<double>(3);
}

TEST_CASE("setop with float") {
    test_setop_with_T<float>(1);
    test_setop_with_T<float>(3);
}

// TEST_CASE("setop with single complex") {
//     test_setop_with_T<complex_single>(1);
//     test_setop_with_T<complex_single>(3);
// }

// TEST_CASE("setop with double complex") {
//     test_setop_with_T<complex_double>(1);
//     test_setop_with_T<complex_double>(3);
// }

TEST_CASE("mapop with double") {
    test_mapop_with_T<double>(1);
    test_mapop_with_T<double>(3);
}

TEST_CASE("mapop with float") {
    test_mapop_with_T<float>(1);
    test_mapop_with_T<float>(3);
}

// TEST_CASE("mapop with single complex") {
//     test_mapop_with_T<complex_single>(1);
//     test_mapop_with_T<complex_single>(3);
// }

// TEST_CASE("mapop with double complex") {
//     test_mapop_with_T<complex_double>(1);
//     test_mapop_with_T<complex_double>(3);
// }

TEST_CASE("addop with double") {
    test_addop_with_T<double>(1);
    test_addop_with_T<double>(3);
}

TEST_CASE("addop with float") {
    test_addop_with_T<float>(1);
    test_addop_with_T<float>(3);
}

// TEST_CASE("addop with single complex") {
//     test_addop_with_T<complex_single>(1);
//     test_addop_with_T<complex_single>(3);
// }

// TEST_CASE("addop with double complex") {
//     test_addop_with_T<complex_double>(1);
//     test_addop_with_T<complex_double>(3);
// }

#if 1
TEST_CASE("Two-dimensional ops") {
    bool failed;
    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};
    using T              = double;

    IndexSpace IS{range(0, 10),
                  {{"nr1", {range(0, 5)}}, {"nr2", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, 1};
    TiledIndexLabel l1, l2, lall, lnone;
    std::tie(l1)   = TIS.labels<1>("nr1");
    std::tie(l2)   = TIS.labels<1>("nr2");
    std::tie(lall) = TIS.labels<1>("all");
    //@todo is there a "none" slice?
    // std::tie(lnone) = TIS.labels<1>("none");
    {
        Tensor<T> T1{TIS, TIS};
        REQUIRE(test_setop(ec, T1, T1(l1, l1)));
    }

    {
        Tensor<T> T1{TIS, TIS};
        REQUIRE(
          test_setop(ec, T1, T1(l1, l1), {T1(l1, l2), T1(l2, l1), T1(l2, l2)}));
    }

    {
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS};
        test_addop(ec, T1, T2, T1(), T2());
    }
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() += T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() += 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 9 + 1.5 * 8 * 4);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() += 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value(T3, 4 + 1.5 * 100 * 9 * 8);
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);
    //MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}

TEST_CASE("Two-dimensional ops with flush and sync") {
    bool failed;
    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};
    using T              = double;

    IndexSpace IS{range(0, 10),
                  {{"nr1", {range(0, 5)}}, {"nr2", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, 1};
    TiledIndexLabel l1, l2, lall, lnone;
    std::tie(l1)   = TIS.labels<1>("nr1");
    std::tie(l2)   = TIS.labels<1>("nr2");
    std::tie(lall) = TIS.labels<1>("all");
    //@todo is there a "none" slice?
    // std::tie(lnone) = TIS.labels<1>("none");
    {
        Tensor<T> T1{TIS, TIS};
        REQUIRE(test_setop(ec, T1, T1(l1, l1)));
    }

    {
        Tensor<T> T1{TIS, TIS};
        REQUIRE(
          test_setop(ec, T1, T1(l1, l1), {T1(l1, l2), T1(l2, l1), T1(l2, l2)}));
    }

    {
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS};
        test_addop(ec, T1, T2, T1(), T2());
    }
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() += T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)32.0);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() += 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 9 + 1.5 * 8 * 4);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() += 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value(T3, 4 + 1.5 * 100 * 9 * 8);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);
    
    ec->flush_and_sync();
    
    //MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}

TEST_CASE("One-dimensional ops") {
    bool failed;
    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};
    using T              = double;

    IndexSpace IS{range(0, 10),
                  {{"nr1", {range(0, 5)}}, {"nr2", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, 1};
    TiledIndexLabel l1, l2, lall, lnone;
    std::tie(l1)   = TIS.labels<1>("nr1");
    std::tie(l2)   = TIS.labels<1>("nr2");
    std::tie(lall) = TIS.labels<1>("all");
    //@todo is there a "none" slice?
    // std::tie(lnone) = TIS.labels<1>("none");

    {
        Tensor<T> T1{TIS};
        REQUIRE(test_setop(ec, T1, T1()));
    }

    {
        Tensor<T> T1{TIS};
        REQUIRE(test_setop(ec, T1, T1(l1), {T1(l2)}));
    }

    //@todo slice addop tests
    {
        Tensor<T> T1{TIS}, T2{TIS};
        test_addop(ec, T1, T2, T1(), T2());
    }

    try {
        failed = false;
        Tensor<T> T1{TIS}, T2{TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() += T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS}, T2{TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() += 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 9 + 1.5 * 8 * 4);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS}, T2{TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() += 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value(T3, 4 + 1.5 * 10 * 9 * 8);
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);
    //MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}
#endif

TEST_CASE("Three-dimensional mult ops part I") {
    bool failed;
    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};
    using T               = double;
    const size_t tilesize = 1;

    IndexSpace IS{range(0, 10),
                  {{"nr1", {range(0, 5)}}, {"nr2", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, tilesize};
    TiledIndexLabel i, j, k, l;
    std::tie(i, j, k, l) = TIS.labels<4>("all");

#if 1
    // mult 3x3x0
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS}, T2{TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 2)(T2() = 3)(T3() = 4)(T1() += 6.9 * T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(2 + 6.9 * 3 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    // mult 3x0x3
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS}, T2{TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 2)(T2() = 3)(T3() = 4)(T1() += 1.7 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(2 + 1.7 * 3 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    // mult 3x2x1
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS}, T2{TIS, TIS}, T3{TIS};
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 2)(T2() = 3)(T3() = 4)(
            T1(i, j, k) += 1.7 * T2(i, j) * T3(k))
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(2 + 1.7 * 3 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);
#endif

    // mult 3x3x3
#if 1
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS}, T2{TIS, TIS, TIS}, T3{TIS, TIS, TIS};
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 2)(T2() = 3)(T3() = 4)(
            T1(i, j, k) += 1.7 * T2(j, l, i) * T3(l, i, k))
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(2 + 1.7 * 3 * 4 * 10));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);
#endif
    //MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}

TEST_CASE("Four-dimensional mult ops part I") {
    bool failed;
    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};
    using T               = double;
    const size_t tilesize = 1;

    IndexSpace IS{range(0, 10),
                  {{"nr1", {range(0, 5)}}, {"nr2", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, tilesize};
    TiledIndexLabel i, j, k, l, m, n;
    std::tie(i, j, k, l, m, n) = TIS.labels<6>("all");

    // mult 4x4x0
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS, TIS}, T2{TIS, TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 2)(T2() = 3)(T3() = 4)(T1() += 6.9 * T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(2 + 6.9 * 3 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    // mult 4x0x4
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS, TIS}, T2{TIS, TIS, TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 2)(T2() = 3)(T3() = 4)(T1() += 1.7 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(2 + 1.7 * 3 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    // mult 4x2x2
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS, TIS}, T2{TIS, TIS}, T3{TIS, TIS};
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 2)(T2() = 3)(T3() = 4)(
            T1(i, j, k, l) += 1.7 * T2(i, j) * T3(k, l))
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(2 + 1.7 * 3 * 4));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    // mult 4x4x4
#if 1
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS, TIS, TIS}, T2{TIS, TIS, TIS, TIS},
          T3{TIS, TIS, TIS, TIS};
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 2)(T2() = 3)(T3() = 4)(
            T1(i, j, k, l) += 1.7 * T2(j, l, k, m) * T3(l, i, k, m))
          .deallocate(T2, T3)
          .execute();
        check_value(T1, (T)(2 + 1.7 * 3 * 4 * 10));
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);
#endif
    //MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}

#if 1
TEST_CASE("Two-dimensional ops part I") {
    bool failed;
    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};
    using T              = double;

    IndexSpace IS{range(0, 10),
                  {{"nr1", {range(0, 5)}}, {"nr2", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, 1};

    // setop
    {
        Tensor<T> T1{TIS, TIS};
        Tensor<T>::allocate(ec, T1);
        Scheduler{*ec}(T1() = 42).execute();
        check_value(T1, (T)42.0);
        Tensor<T>::deallocate(T1);
    }

    // addop
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS};
        Scheduler{*ec}
          .allocate(T1, T2)(T2() = 42)(T1() = T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)42.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);
    
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS};
        Scheduler{*ec}
          .allocate(T1, T2)(T1() = 4)(T2() = 42)(T1() += T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)46.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS};
        Scheduler{*ec}
          .allocate(T1, T2)(T1() = 4)(T2() = 42)(T1() += 3 * T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)130.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS};
        Scheduler{*ec}
          .allocate(T1, T2)(T1() = 4)(T2() = 42)(T1() -= T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)-38.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS};
        Scheduler{*ec}
          .allocate(T1, T2)(T1() = 4)(T2() = 42)(T1() += -3.1 * T2())
          .deallocate(T2)
          .execute();
        check_value(T1, (T)-126.2);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    // multop: 2,2,0
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 4)(T2() = 42)(T3() = 5)(
            T1() += -3.1 * T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 4 - 3.1 * 42 * 5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 4)(T2() = 42)(T3() = 5)(
            T1() -= -3.1 * T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 4 + 3.1 * 42 * 5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{}, T2{TIS, TIS}, T3{TIS, TIS};
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 4)(T2() = 42)(T3() = 5)(
            T1() += -3.1 * T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 4 - 3.1 * 10 * 10 * 42 * 5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    // multop 2,1,1
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS}, T3{TIS};
        TiledIndexLabel i, j;
        std::tie(i, j) = TIS.labels<2>("all");
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 4)(T2() = 42)(T3() = 5)(
            T1(i, j) += -3.1 * T2(i) * T3(j))
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 4 - 3.1 * 42 * 5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    // multop 2,2,1
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{TIS};
        TiledIndexLabel i, j;
        std::tie(i, j) = TIS.labels<2>("all");
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 4)(T2() = 42)(T3() = 5)(
            T1(i, j) += -3.1 * T2(i, j) * T3(i))
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 4 - 3.1 * 42 * 5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{TIS};
        TiledIndexLabel i, j;
        std::tie(i, j) = TIS.labels<2>("all");
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 4)(T2() = 42)(T3() = 5)(
            T1(i, j) += -3.1 * T2(j, i) * T3(i))
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 4 - 3.1 * 42 * 5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{TIS};
        TiledIndexLabel i, j;
        std::tie(i, j) = TIS.labels<2>("all");
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 4)(T2() = 42)(T3() = 5)(
            T1(i, j) += -3.1 * T2(i, j) * T3(j))
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 4 - 3.1 * 42 * 5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{TIS};
        TiledIndexLabel i, j;
        std::tie(i, j) = TIS.labels<2>("all");
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 4)(T2() = 42)(T3() = 5)(
            T1(i, j) += -3.1 * T3(j) * T2(j, i))
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 4 - 3.1 * 42 * 5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    // multop 2,2,2
    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{TIS, TIS};
        TiledIndexLabel i, j, k;
        std::tie(i, j, k) = TIS.labels<3>("all");
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 4)(T2() = 42)(T3() = 5)(
            T1(i, j) += -3.1 * T2(i, j) * T3(i, j))
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 4 - 3.1 * 42 * 5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS, TIS}, T2{TIS, TIS}, T3{TIS, TIS};
        TiledIndexLabel i, j, k;
        std::tie(i, j, k) = TIS.labels<3>("all");
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 4)(T2() = 42)(T3() = 5)(
            T1(i, j) += -3.1 * T2(i, k) * T3(j, k))
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 4 - 3.1 * 42 * 5 * 10);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);
}
#endif

TEST_CASE("MultOp with RHS reduction") {
    bool failed;
    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};
    using T              = double;

    IndexSpace IS{range(0, 4),
                  {{"nr1", {range(0, 2)}}, {"nr2", {range(2, 4)}}}};
    TiledIndexSpace TIS{IS, 1};

    try {
        failed = false;
        Tensor<T> T1{}, T2{TIS}, T3{};
        TiledIndexLabel i, j, k;
        std::tie(i, j, k) = TIS.labels<3>("all");
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 4)(T2() = 3)(T3() = 5)(
            T1() += -3.1 * T2(i) * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 4.0 - 3.1 * 4 * 3 * 5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{}, T2{}, T3{TIS};
        TiledIndexLabel i, j, k;
        std::tie(i, j, k) = TIS.labels<3>("all");
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 4)(T2() = 3)(T3() = 5)(
            T1() += -3.1 * T2() * T3(j))
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 4.0 - 3.1 * 4 * 3 * 5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{}, T2{TIS}, T3{TIS};
        TiledIndexLabel i, j, k;
        std::tie(i, j, k) = TIS.labels<3>("all");
        Scheduler{*ec}
          .allocate(T1, T2, T3)(T1() = 4)(T2() = 3)(T3() = 5)(
            T1() += -3.2 * T2(i) * T3(j))
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 4.0 - 3.2 * 4 * 4 * 3 * 5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try{

        failed = false;
        IndexSpace cvec{range(0,4)};
        TiledIndexSpace CV{cvec,1};

        IndexSpace MO_IS{range(0, 14),
                        {{"occ", {range(0, 10)}},
                        {"virt", {range(10,14)}}}};
        TiledIndexSpace MO{MO_IS, {5,5,2,2}};

        TiledIndexSpace O = MO("occ");
        TiledIndexSpace V = MO("virt");
        TiledIndexSpace N = MO("all");

        Tensor<T> CV3D{N,N,CV};
        Tensor<T> CV2D{N,N};
        Tensor<T> res{O, O};
        Tensor<T> res1{O, O};
        Tensor<T> tmp1{O, O};
        Tensor<T> tmp2{O, O};
        Tensor<T> t1{V, O};
        Tensor<T> sc1{};
        Tensor<T> sc2{};

        Tensor<T>::allocate(ec,sc1,sc2,res,res1,tmp1,tmp2,t1,CV3D,CV2D);
        Scheduler sch{*ec};
        
        sch(CV3D() = 42)
        (CV2D() = 42)
        (t1() = 2)
        (res() = 0)
        (res1() = 0)
        (sc1() = 0)
        (sc2() = 0)
        (tmp1() = 0)
        (tmp2() = 0)
        .execute();


        for(const IndexVector& blockid : CV2D.loop_nest()) {
            const TAMM_SIZE size = CV2D.block_size(blockid);
            std::vector<T> buf(size);
            CV2D.get(blockid, buf);

            auto block_dims   = CV2D.block_dims(blockid);
            auto block_offset = CV2D.block_offsets(blockid);

            TAMM_SIZE c = 0;
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                for(size_t j = block_offset[1];
                    j < block_offset[1] + block_dims[1]; j++, c++) {
                       buf[c]=i+j*c;
                }
            }
        }
        
        for(const IndexVector& blockid : CV3D.loop_nest()) {
            const TAMM_SIZE size = CV3D.block_size(blockid);
            std::vector<T> buf(size);
            CV3D.get(blockid, buf);

            auto block_dims   = CV3D.block_dims(blockid);
            auto block_offset = CV3D.block_offsets(blockid);

            TAMM_SIZE c = 0;
            TAMM_SIZE c1 = 0;
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                for(size_t j = block_offset[1];
                    j < block_offset[1] + block_dims[1]; j++, c1++) {
                for(size_t k = block_offset[2];
                    k < block_offset[2] + block_dims[2]; k++, c++) 
                       buf[c]=i+j*c1;
                }
            }
        }

        ec->pg().barrier();

        TiledIndexLabel cind, h1,h2,h3,p1,p2,p3;
        std::tie(cind) = CV.labels<1>("all");
        std::tie(p1, p2,p3) = MO.labels<3>("virt");
        std::tie(h1, h2,h3) = MO.labels<3>("occ");

        sch(res(h2, h1) += 1.0 * t1(p1, h1) * CV3D(h2, p1, cind));
        sch(sc1() += 1.0 * t1(p1, h1) * CV3D(h1, p1, cind));
        //sch(tmp1(h2, h1) += -1.0 * res(h2, h1) * sc1());
        sch(tmp1(h2, h1) += -1.0 * res(h3, h1) * res(h2,h3));

        for (auto ci=0;ci<4;ci++){
            sch(res1(h2, h1) += 1.0 * t1(p1, h1) * CV2D(h2, p1));
            sch(sc2() += 1.0 * t1(p1, h1) * CV2D(h1, p1));
        }
        //sch(tmp2(h2, h1) += -1.0 * res1(h3, h1) * sc2());
        sch(tmp2(h2, h1) += -1.0 * res1(h3, h1) * res1(h2,h3));


        sch.execute();

        for(const IndexVector& blockid : tmp2.loop_nest()) {
            const TAMM_SIZE size = tmp2.block_size(blockid);
            std::vector<T> buf(size);
            tmp2.get(blockid, buf);

            std::vector<T> buf1(size);
            tmp1.get(blockid, buf1);
            auto block_dims   = tmp2.block_dims(blockid);
            auto block_offset = tmp2.block_offsets(blockid);

            TAMM_SIZE c = 0;
            for(size_t i = block_offset[0]; i < block_offset[0] + block_dims[0];
                i++) {
                for(size_t j = block_offset[1];
                    j < block_offset[1] + block_dims[1]; j++, c++) {
                        REQUIRE(buf[c] == buf1[c]);
                }
            }
        }
        
        ec->pg().barrier();

        // T r1, r2;
        // sc1.get({}, {&r1, 1});
        // sc2.get({}, {&r2, 1});

        // REQUIRE(r1==r2);

        Tensor<T>::deallocate(sc1,sc2,res,res1,tmp1,tmp2,t1,CV3D,CV2D);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);
}
