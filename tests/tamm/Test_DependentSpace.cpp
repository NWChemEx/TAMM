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
        t.get(it,span<T>(&buf[0],size));
        std::cout << "block" << it;
        for (TAMM_SIZE i = 0; i < size;i++)
         std::cout << i << std::endl;
    }
}

template<typename T>
void check_value(Tensor<T> &t, T val){
    for (auto it: t.loop_nest())
    {
        TAMM_SIZE size = t.block_size(it);
        std::vector<T> buf(size);
        t.get(it,span<T>(&buf[0],size));
        for (TAMM_SIZE i = 0; i < size;i++) {
          REQUIRE(std::fabs(buf[i]-val)< 1.0e-10);
       }
    }
}

template<typename T>
void check_value(LabeledTensor<T> lt, T val){
    // std::cerr << __FUNCTION__ << " " << __LINE__ << "\n";
    Tensor<T> t = lt.tensor();
    IndexLabelVec unique_labels = internal::unique_entries(lt.labels());
    
    const IndexLabelVec& sorted_labels = internal::sort_on_dependence(unique_labels);

    std::vector<IndexLoopBound> ilbs;
    for(const auto& lbl: sorted_labels) {
        ilbs.push_back(lbl);
    }
    IndexLoopNest loop_nest{ilbs};

    const std::vector<size_t>& lhs_pm =
        internal::perm_map_compute(sorted_labels, lt.labels());

    for (const auto& it: loop_nest)
    {
        const IndexVector& blockid =
            internal::perm_map_apply(it, lhs_pm);
        size_t size = t.block_size(blockid);
        std::vector<T> buf(size);
        t.get(blockid, span<T>(&buf[0],size));
        for (TAMM_SIZE i = 0; i < size; i++) {
          REQUIRE(std::fabs(buf[i]-val)< 1.0e-10);
       }
    }
}

template<typename T>
bool test_setop(ExecutionContext* ec, Tensor<T> T1, LabeledTensor<T> LT1,
                const std::vector<LabeledTensor<T>>& rest_lts = {}) {
    bool success = true;
    try {
        Tensor<T>::allocate(ec, T1);
        try {
            Scheduler{ec}(T1() = -1.0)(LT1 = 42).execute();
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
        Scheduler{ec}(T1() = -1.0)(LT2 = 42)(LT1 = LT2).execute();
        check_value(LT1, (T)42.0);
        for(const auto& lt : rest_lts) { check_value(lt, (T)-1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 0. Exception: " << e << "\n";
        success = false;
    }

    try {
        success = true;
        Scheduler{ec}(T1() = -1.0)(LT1 = 4)(LT2 = 42)(LT1 += LT2).execute();
        check_value(LT1, (T)46.0);
        for(const auto& lt : rest_lts) { check_value(lt, (T)-1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 1. Exception: " << e << "\n";
        success = false;
    }

    try {
        success = true;
        Scheduler{ec}(T1() = -1.0)(LT1 = 4)(LT2 = 42)(LT1 += 3 * LT2)
          .execute();
        check_value(T1, (T)130.0);
        for(const auto& lt : rest_lts) { check_value(lt, (T)-1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 2. Exception: " << e << "\n";
        success = false;
    }

    try {
        success = true;
        Scheduler{ec}(T1() = -1.0)(T1() = 4)(T2() = 42)(T1() -= T2()).execute();
        check_value(T1, (T)-38.0);
        for(const auto& lt : rest_lts) { check_value(lt, (T)-1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 3. Exception: " << e << "\n";
        success = false;
    }

    try {
        success = true;
        Scheduler{ec}
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

    ProcGroup pg{GA_MPI_Comm()};
    MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};

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
    ProcGroup pg{GA_MPI_Comm()};
    MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};

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
        Scheduler{ec}
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
        Scheduler{ec}
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
        Scheduler{ec}
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
        Scheduler{ec}
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
        Scheduler{ec}
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
        Scheduler{ec}
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
        Scheduler{ec}
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
        Scheduler{ec}
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
        Scheduler{ec}
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
        Scheduler{ec}
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
        Scheduler{ec}
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
        Scheduler{ec}
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
        Scheduler{ec}
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
        Scheduler{ec}
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
        Scheduler{ec}
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

TEST_CASE("Two-dimensional ops on dependent space") {
    bool success = false;
    ProcGroup pg{GA_MPI_Comm()};
    MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};
    using T              = double;

    IndexSpace IS{range(0, 10)};
    TiledIndexSpace T_IS{IS};
    std::map<IndexVector, IndexSpace> dep_relation{
        {IndexVector{0}, IS},
        {IndexVector{1}, IS},
        {IndexVector{2}, IS},
        {IndexVector{3}, IS},
        {IndexVector{4}, IS},
        {IndexVector{5}, IS},
        {IndexVector{6}, IS},
        {IndexVector{7}, IS},
        {IndexVector{8}, IS},
        {IndexVector{9}, IS}
    };

    IndexSpace DIS{{T_IS}, dep_relation};

    TiledIndexSpace T_DIS{DIS};

    TiledIndexLabel a, i;

    std::tie(a) = T_DIS.labels<1>("all");
    std::tie(i) = T_IS.labels<1>("all");

    
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

    // Tensor Allocation 
    {
        success = true;
        try
        {
            Tensor<T> T1{a(i), i};
            Tensor<T>::allocate(ec, T1);
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
            Scheduler{ec}(T1() = 42).execute();
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

    // SetOp test with zero labels
    {
        Tensor<T> T1{a(i), i};
        REQUIRE(test_setop(ec, T1, T1()));
    }

    // SetOp test with labels provided 
    {
        Tensor<T> T1{a(i), i};
        REQUIRE(test_setop(ec, T1, T1(a(i),i)));
    }

    TiledIndexSpace Sub_TIS1{T_IS, range(0,5)};
    TiledIndexSpace Sub_TIS2{T_IS, range(5,10)};

    std::map<IndexVector, TiledIndexSpace> sub_relation1{
        {IndexVector{0}, Sub_TIS1},
        {IndexVector{1}, Sub_TIS1},
        {IndexVector{2}, Sub_TIS1},
        {IndexVector{3}, Sub_TIS1},
        {IndexVector{4}, Sub_TIS1}
    };

    std::map<IndexVector, TiledIndexSpace> sub_relation2{
        {IndexVector{5}, Sub_TIS2},
        {IndexVector{6}, Sub_TIS2},
        {IndexVector{7}, Sub_TIS2},
        {IndexVector{8}, Sub_TIS2},
        {IndexVector{9}, Sub_TIS2}
    };
    // Creating sub tiled spaces Dependent-TiledIndexSpace
    {
        success = true;
        try
        {
            TiledIndexSpace SUB_TDIS1{T_DIS, sub_relation1};
            TiledIndexSpace SUB_TDIS2{T_DIS, sub_relation2};            
        }
        catch(const std::string& e)
        {
            std::cerr << "Caught exception: " << e << "\n";
            success = false;
        }
        REQUIRE(success);
    }

}