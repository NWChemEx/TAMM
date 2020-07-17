#define CATCH_CONFIG_RUNNER
#include "catch/catch.hpp"
#include "ga.h"
#include "mpi.h"
#include "macdecls.h"
#include "ga-mpi.h"
#include "tamm/tamm.hpp"

#include <string>

using namespace tamm;


template<typename T>
void print_tensor(Tensor<T> &t){
    std::cout << "Print tensor: " << &t << std::endl;
    for (auto it: t.loop_nest())
    {
        TAMM_SIZE size = t.block_size(it);
        std::vector<T> buf(size);
        t.get(it, buf);
        std::cout << "block" << it;
        for (TAMM_SIZE i = 0; i < size;i++)
         std::cout << i << " ";
        std::cout << std::endl;
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
            REQUIRE(std::fabs(buf[i] - val) < 1.0e-10);
        }
    }
}

template<typename T>
void check_value(Tensor<T>& t, T val) {
    check_value(t(), val);
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
        Scheduler{*ec}(T1() = -1.0)(LT1 = 4)(LT2 = 42)(LT1 += 3 * LT2)
          .execute();
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
        Scheduler{*ec}
          (T1() = -1.0)(T1() = 4)(T2() = 42)(T1() += -3.1 * T2())
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

//setop with T (call with tilesize 1 and 3)
template<typename T>
void test_setop_with_T(unsigned tilesize) {
    //0-4 dimensional setops
    //0-4 dimensional setops

    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg,DistributionKind::nw, MemoryManagerKind::ga};

    IndexSpace IS{range(0, 10),
                  {{"nr1", {range(0, 5)}}, {"nr2", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, tilesize};
    TiledIndexLabel l1, l2, lall, lnone;
    std::tie(l1) = TIS.labels<1>("nr1");
    std::tie(l2) = TIS.labels<1>("nr2");
    std::tie(lall) = TIS.labels<1>("all");
    //@todo is there a "none" slice?
    //std::tie(lnone) = TIS.labels<1>("none");

    //0-dimensional test
    {
        Tensor<T> T1{};
        REQUIRE(test_setop(ec, T1, T1()));
    }

    //1-dimensional tests
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


    //2-dimensional tests
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
        REQUIRE(test_setop(ec, T1, T1(l1, l1), {T1(l1, l2), T1(l2, l1), T1(l2, l2)}));
    }
    //3-dimensional tests
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
        REQUIRE(test_setop(ec, T1, T1(l1, l2, l2), {T1(l1, l1, l1), T1(l1, l1, l2), T1(l1, l2, l1), T1(l2, l1, l1), T1(l2, l1, l2), T1(l2, l2, l1), T1(l2, l2, l2)}));
    }
    //4-dimensional tests
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
        REQUIRE(test_setop(ec, T1, T1(l1, l2, l2, l1), {T1(l1, l1, l1, l1), T1(l1, l1, l1, l2), T1(l1, l1, l2, l1), T1(l1, l1, l2, l2), T1(l1, l2, l1, l1), T1(l1, l2, l1, l2), T1(l1, l2, l2, l2), T1(l2, l1, l1, l1), T1(l2, l1, l1, l2), T1(l2, l1, l2, l1), T1(l2, l2, l1, l1), T1(l2, l2, l1, l2), T1(l2, l2, l2, l1), T1(l2, l2, l2, l2)}));
   }

    MemoryManagerGA::destroy_coll(mgr);
    delete ec;

}

//addop with T  (call with tilesize 1 and 3)
template<typename T>
void test_addop_with_T(unsigned tilesize) {
    //0-4 dimensional addops
    bool failed;
    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};

    IndexSpace IS{range(0, 10),
                  {{"nr1", {range(0, 5)}}, {"nr2", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, tilesize};
    //0-dimensional test
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

    //1-dimension tests
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

    //2-dimension tests
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
 
    //3-dimension tests
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
 
    //4-dimension tests
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
 
    MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}


template<typename T> 
void test_dependent_space_with_T(Index tilesize) {
    bool success = false;
    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext* ec = new ExecutionContext{pg, DistributionKind::nw, MemoryManagerKind::ga};

    IndexSpace IS{range(0, 10)};
    TiledIndexSpace T_IS{IS, tilesize};

    std::map<IndexVector, IndexSpace> dep_relation;

    Index tile_count = (10 / tilesize);
    if((10 % tilesize) > 0){ tile_count++; }

    for (Index i = 0; i < tile_count; i++) {
        dep_relation.insert({IndexVector{i}, IS});
    }

    IndexSpace DIS{{T_IS}, dep_relation};

    TiledIndexSpace T_DIS{DIS, tilesize};

    TiledIndexLabel a, b, i, j;

    std::tie(a, b) = T_DIS.labels<2>("all");
    std::tie(i, j) = T_IS.labels<2>("all");

    // 2-dimensional tests
    // Tensor Construction 
    {
        success = true;
        try
        {
            Tensor<T> T1{a(i), i};
        }
        catch(const std::string & e)
        {
            std::cerr << "Caught exception: " << e << "\n";
            success = false;
        }
        REQUIRE(success);
    }

    // Tensor Allocation / Deallocate
    {
        success = true;
        try
        {
            Tensor<T> T1{a(i), i};
            Tensor<T>::allocate(ec, T1);
            Tensor<T>::deallocate(T1);
        }
        catch(const std::string & e)
        {
            std::cerr << "Caught exception: " << e << "\n";
            success = false;
        }
        REQUIRE(success);
    }
    
    // Basic SetOp 
    {
        success = true;
        try
        {
            Tensor<T> T1{a(i), i};
            Tensor<T>::allocate(ec, T1);
            Scheduler{*ec}(T1() = 42).execute();
            check_value(T1, (T)42.0);
            Tensor<T>::deallocate(T1);
        }
        catch(const std::string & e)
        {
            std::cerr << "Caught exception: " << e << "\n";
            success = false;
        }
        REQUIRE(success);
    }

    // SetOp test with no labels
    {
        Tensor<T> T1{a(i), i};
        REQUIRE(test_setop(ec, T1, T1()));
    }

    // SetOp test with labels provided 
    {
        Tensor<T> T1{a(i), i};
        REQUIRE(test_setop(ec, T1, T1(a(i),i)));
    }

    // AddOp test with no labels
    {
        Tensor<T> T1{a(i), i};
        Tensor<T> T2{a(i), i};
        REQUIRE(test_addop(ec, T1, T2, T1(), T2()));
    }

#if 0
    // AddOp test with no labels on rhs
    {
        Tensor<T> T1{a(i), i};
        Tensor<T> T2{a(i), i};
        REQUIRE(test_addop(ec, T1, T2, T1(a(i), i), T2()));
    }

    // AddOp test with no labels in lhs
    {
        Tensor<T> T1{a(i), i};
        Tensor<T> T2{a(i), i};
        REQUIRE(test_addop(ec, T1, T2, T1(), T2(a(i), i)));
    }
#endif
    // AddOp test with labels
    {
        Tensor<T> T1{a(i), i};
        Tensor<T> T2{a(i), i};
        REQUIRE(test_addop(ec, T1, T2, T1(a(i), i), T2(a(i), i)));
    }

    // MultOp 2-dim += 2-dim * 0-dim
    try {
        success = true;
        Tensor<T> T1{a(i), i}, T2{a(i), i}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() += T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value<T>(T1, (T)32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        success = false;
    }
    REQUIRE(success);

    // MultOp 2-dim += alpha * 0-dim * 2-dim 
    try {
        success = true;
        Tensor<T> T1{a(i), i}, T2{a(i), i}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() += 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value<T>(T1, 9 + 1.5 * 8 * 4);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        success = false;
    }
    REQUIRE(success);

    // MultOp 0-dim += alpha * 2-dim * 2-dim 
    try {
        success = true;
        Tensor<T> T1{a(i), i}, T2{a(i), i}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() += 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value<T>(T3, 4 + 1.5 * 100 * 9 * 8);
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        success = false;
    }
    REQUIRE(success);

    ///////////////////////////////////////////////////////////////
    
    // 3-dimensional tests
    // Tensor Construction 
    {
        success = true;
        try
        {
            Tensor<T> T1{a(i), i, j};
        }
        catch(const std::string & e)
        {
            std::cerr << "Caught exception: " << e << "\n";
            success = false;
        }
        REQUIRE(success);
    }

    // Tensor Allocation / Deallocate
    {
        success = true;
        try
        {
            Tensor<T> T1{a(i), i, j};
            Tensor<T>::allocate(ec, T1);
            Tensor<T>::deallocate(T1);
        }
        catch(const std::string & e)
        {
            std::cerr << "Caught exception: " << e << "\n";
            success = false;
        }
        REQUIRE(success);
    }
    
    // Basic SetOp 
    {
        success = true;
        try
        {
            Tensor<T> T1{a(i), i, j};
            Tensor<T>::allocate(ec, T1);
            Scheduler{*ec}(T1() = 42).execute();
            check_value(T1, (T)42.0);
            Tensor<T>::deallocate(T1);
        }
        catch(const std::string & e)
        {
            std::cerr << "Caught exception: " << e << "\n";
            success = false;
        }
        REQUIRE(success);
    }

    // SetOp test with no labels
    {
        Tensor<T> T1{a(i), i, j};
        REQUIRE(test_setop(ec, T1, T1()));
    }

    // SetOp test with labels provided 
    {
        Tensor<T> T1{a(i), i, j};
        REQUIRE(test_setop(ec, T1, T1(a(i), i, j)));
    }

    // AddOp test with no labels
    {
        Tensor<T> T1{a(i), i, j};
        Tensor<T> T2{a(i), i, j};
        REQUIRE(test_addop(ec, T1, T2, T1(), T2()));
    }
#if 0    
    // AddOp test with no labels on rhs
    {
        Tensor<T> T1{a(i), i, j};
        Tensor<T> T2{a(i), i, j};
        REQUIRE(test_addop(ec, T1, T2, T1(a(i), i, j), T2()));
    }

    // AddOp test with no labels in lhs
    {
        Tensor<T> T1{a(i), i, j};
        Tensor<T> T2{a(i), i, j};
        REQUIRE(test_addop(ec, T1, T2, T1(), T2(a(i), i, j)));
    }
#endif
    // AddOp test with labels
    {
        Tensor<T> T1{a(i), i, j};
        Tensor<T> T2{a(i), i, j};
        REQUIRE(test_addop(ec, T1, T2, T1(a(i), i, j), T2(a(i), i, j)));
    }

    // MultOp 3-dim += 3-dim * 0-dim
    try {
        success = true;
        Tensor<T> T1{a(i), i, j}, T2{a(i), i, j}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() += T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value<T>(T1, (T)32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        success = false;
    }
    REQUIRE(success);

    // MultOp 3-dim += alpha * 0-dim * 3-dim 
    try {
        success = true;
        Tensor<T> T1{a(i), i, j}, T2{a(i), i, j}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() += 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value<T>(T1, 9 + 1.5 * 8 * 4);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        success = false;
    }
    REQUIRE(success);

    std::cerr << "/* message */"	<< std::endl;
    // MultOp 0-dim += alpha * 3-dim * 3-dim 
    try {
        success = true;
        Tensor<T> T1{a(i), i, j}, T2{a(i), i, j}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() += 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value<T>(T3, 4 + 1.5 * 1000 * 9 * 8);
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        success = false;
    }
    REQUIRE(success);

    std::cerr << "Finished default dependent space"	<< std::endl;

    ///////////////////////////////////////////////////////////////
    
    // 4-dimensional tests
    // Tensor Construction 
    {
        success = true;
        try
        {
            Tensor<T> T1{a(i), i, b(j), j};
        }
        catch(const std::string & e)
        {
            std::cerr << "Caught exception: " << e << "\n";
            success = false;
        }
        REQUIRE(success);
    }

    // Tensor Allocation / Deallocate
    {
        success = true;
        try
        {
            Tensor<T> T1{a(i), i, b(j), j};
            Tensor<T>::allocate(ec, T1);
            Tensor<T>::deallocate(T1);
        }
        catch(const std::string & e)
        {
            std::cerr << "Caught exception: " << e << "\n";
            success = false;
        }
        REQUIRE(success);
    }
    
    // Basic SetOp 
    {
        success = true;
        try
        {
            Tensor<T> T1{a(i), i, b(j), j};
            Tensor<T>::allocate(ec, T1);
            Scheduler{*ec}(T1() = 42).execute();
            check_value(T1, (T)42.0);
            Tensor<T>::deallocate(T1);
        }
        catch(const std::string & e)
        {
            std::cerr << "Caught exception: " << e << "\n";
            success = false;
        }
        REQUIRE(success);
    }

    // SetOp test with no labels
    {
        Tensor<T> T1{a(i), i, j};
        REQUIRE(test_setop(ec, T1, T1()));
    }

    // SetOp test with labels provided 
    {
        Tensor<T> T1{a(i), i, b(j), j};
        REQUIRE(test_setop(ec, T1, T1(a(i), i, b(j), j)));
    }

    // AddOp test with no labels
    {
        Tensor<T> T1{a(i), i, b(j), j};
        Tensor<T> T2{a(i), i, b(j), j};
        REQUIRE(test_addop(ec, T1, T2, T1(), T2()));
    }

#if 0
    // AddOp test with no labels on rhs
    {
        Tensor<T> T1{a(i), i, b(j), j};
        Tensor<T> T2{a(i), i, b(j), j};
        REQUIRE(test_addop(ec, T1, T2, T1(a(i), i, b(j), j), T2()));
    }

    // AddOp test with no labels in lhs
    {
        Tensor<T> T1{a(i), i, b(j), j};
        Tensor<T> T2{a(i), i, b(j), j};
        REQUIRE(test_addop(ec, T1, T2, T1(), T2(a(i), i, b(j), j)));
    }
#endif
    // AddOp test with labels
    {
        Tensor<T> T1{a(i), i, b(j), j};
        Tensor<T> T2{a(i), i, b(j), j};
        REQUIRE(test_addop(ec, T1, T2, T1(a(i), i, b(j), j), T2(a(i), i, b(j), j)));
    }

    // MultOp 4-dim += 4-dim * 0-dim
    try {
        success = true;
        Tensor<T> T1{a(i), i, b(j), j}, T2{a(i), i, b(j), j}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 0)(T2() = 8)(T3() = 4)(T1() += T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value<T>(T1, (T)32.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        success = false;
    }
    REQUIRE(success);

    // MultOp 4-dim += alpha * 0-dim * 4-dim 
    try {
        success = true;
        Tensor<T> T1{a(i), i, b(j), j}, T2{a(i), i, b(j), j}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T1() += 1.5 * T3() * T2())
          .deallocate(T2, T3)
          .execute();
        check_value<T>(T1, 9 + 1.5 * 8 * 4);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        success = false;
    }
    REQUIRE(success);

    // MultOp 0-dim += alpha * 3-dim * 3-dim 
    try {
        success = true;
        Tensor<T> T1{a(i), i, b(j), j}, T2{a(i), i, b(j), j}, T3{};
        Scheduler{*ec}
          .allocate(T1, T2,
                    T3)(T1() = 9)(T2() = 8)(T3() = 4)(T3() += 1.5 * T1() * T2())
          .deallocate(T1, T2)
          .execute();
        check_value<T>(T3, 4 + 1.5 * 10000 * 9 * 8);
        Tensor<T>::deallocate(T3);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        success = false;
    }
    REQUIRE(success);

    std::cerr << "Finished default dependent space"	<< std::endl;

    ///////////////////////////////////////////////////////////////

    MemoryManagerGA::destroy_coll(mgr);
    delete ec;

}


TEST_CASE("Tensor ops for double") {
    test_dependent_space_with_T<double>(1);
    test_dependent_space_with_T<double>(3);
}

TEST_CASE("Tensor ops for float") {
    test_dependent_space_with_T<float>(1);
    test_dependent_space_with_T<float>(3);
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
