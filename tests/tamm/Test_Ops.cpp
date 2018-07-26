//#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_RUNNER
#include "catch/catch.hpp"
#include "ga.h"
#include "mpi.h"
#include "macdecls.h"
#include "ga-mpi.h"
#include "tamm/tamm.hpp"

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
        T* buf = new T[size];
        t.get(it,span<T>(buf,size));
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
        T* buf = new T[size];
        t.get(it,span<T>(buf,size));
        for (TAMM_SIZE i = 0; i < size;i++) {
          REQUIRE(std::fabs(buf[i]-val)< 1.0e-10);
       }
    }
}

template<typename T>
void check_value(LabeledTensor<T> lt, T val){
    Tensor<T> t = lt.tensor();
    std::vector<IndexLoopBound> ilbs;
    for(const auto& lbl: lt.labels()) {
        ilbs.push_back(lbl);
    }
    IndexLoopNest loop_nest{ilbs};
    for (const auto& it: loop_nest)
    {
        TAMM_SIZE size = t.block_size(it);
        T* buf = new T[size];
        t.get(it,span<T>(buf,size));
        for (TAMM_SIZE i = 0; i < size;i++) {
          REQUIRE(std::fabs(buf[i]-val)< 1.0e-10);
       }
    }
}

template<typename T>
void test_ops(const TiledIndexSpace& MO) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    const TiledIndexSpace& N = MO("all");

    Tensor<T> d_t1{V, O};
    Tensor<T> d_t2{V, V, O, O};

    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext *ec = new ExecutionContext{pg,&distribution,mgr};

    Tensor<T> T1{N,N,N};
    Tensor<T>::allocate(ec, T1);

    //@todo cleanup this file, seperate get,put,setop tests
    T cx=42;
    size_t size = 1000;
    T* buf = new T[size];
    for(size_t i=0;i<size;i++)
      buf[i]=cx++;

    T1.put(IndexVector{1,0,1}, span<T>(buf,size));

    T* gbuf = new T[size];
    T1.get(IndexVector{1,0,1}, span<T>(gbuf,size));
    for(size_t i=0;i<T1.block_size({1,0,1});i++)
        EXPECTS(gbuf[i]==buf[i]);
    Tensor<T>::deallocate(T1);
    Tensor<T> xt1{N,N};
    Tensor<T> xt2{N,N};
    Tensor<T> xt3{N,N};
   // Tensor<T>::allocate(ec,xt1,xt2,xt3);
  
#if 1
    Scheduler{ec}.allocate(xt1,xt2,xt3)
        (xt1("n1","n2") = 2.2)
        (xt2("n1","n2") = 2.0*xt1("n1","n2"))
        //(xt3("n1","n2") = 2.0*xt1("n1","nk")*xt2("nk","n2")) //no-op
        //.deallocate(xt3)
        .execute();

    check_value(xt1,2.2);
    check_value(xt2,4.4);
#endif

    Tensor<T>::deallocate(xt1,xt2,xt3);
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

// TEST_CASE("Test Ops") {
//     // Construction of tiled index space MO from sketch
//     IndexSpace MO_IS{range(0, 10),
//                      {{"occ", {range(0, 5)}}, {"virt", {range(5, 10)}}}};
//     TiledIndexSpace MO{MO_IS, 1};

//     CHECK_NOTHROW(test_ops<double>(MO));
// }

TEST_CASE("Zero-dimensional ops") {
    ProcGroup pg{GA_MPI_Comm()};
    MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};
    using T              = double;

    IndexSpace IS{range(0, 10)};
    TiledIndexSpace TIS{IS, 1};

    {
        Tensor<T> T1{};
        Tensor<T>::allocate(ec, T1);
        Scheduler{ec}(T1() = 42).execute();
        check_value(T1, 42.0);
        Tensor<T>::deallocate(T1);
    }

    {
        Tensor<T> T1{},T2{};
        Tensor<T>::allocate(ec, T1, T2);
        auto lambda = [](Tensor<T>& t, const IndexVector& iv, std::vector<T>& buf) {
            std::cout << "hi" <<  std::endl;
        };

        Scheduler{ec}(T2() = 42)(T1() = T2()).execute();
        Scheduler{ec}.gop(T1(),lambda).execute();
        //Scheduler{ec}.gop(T1(),std::array{T2()},lambda).execute();
        check_value(T1, 42.0);
        Tensor<T>::deallocate(T1, T2);
    }

    {
        Tensor<T> T1{}, T2{};
        Scheduler{ec}
          .allocate(T1, T2)(T2() = 42)(T1() = T2())
          .deallocate(T2)
          .execute();
        check_value(T1, 42.0);
        Tensor<T>::deallocate(T1);
    }

    {
        Tensor<T> T1{}, T2{};
        Scheduler{ec}
          .allocate(T1, T2)(T1()=3)(T2() = 42)(T1() += T2())
          .deallocate(T2)
          .execute();
        check_value(T1, 45.0);
        Tensor<T>::deallocate(T1);
    }

    {
        Tensor<T> T1{}, T2{};
        Scheduler{ec}
          .allocate(T1, T2)(T1()=42)(T2() = 3)(T1() += 2.5*T2())
          .deallocate(T2)
          .execute();
        check_value(T1, 49.5);
        Tensor<T>::deallocate(T1);
    }

    {
        Tensor<T> T1{}, T2{};
        Scheduler{ec}
          .allocate(T1, T2)(T1()=42)(T2() = 3)(T1() -= T2())
          .deallocate(T2)
          .execute();
        check_value(T1, 39.0);
        Tensor<T>::deallocate(T1);
    }

    {
        Tensor<T> T1{}, T2{};
        Scheduler{ec}
          .allocate(T1, T2)(T1()=42)(T2() = 3)(T1() -= 4*T2())
          .deallocate(T2)
          .execute();
        check_value(T1, 30.0);
        Tensor<T>::deallocate(T1);
    }

    {
        Tensor<T> T1{}, T2{},T3{};
        Scheduler{ec}
          .allocate(T1, T2, T3)(T1()=0)(T2() = 3)(T3() = 5)
          (T1() += T2() * T3())
          .deallocate(T2, T3)
          .execute();
        check_value(T1, 15.0);
        Tensor<T>::deallocate(T1);
    }

    MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}

template<typename T>
bool test_setop(ExecutionContext* ec, Tensor<T> T1, LabeledTensor<T> LT1,
                const std::vector<LabeledTensor<T>>& rest_lts = {}) {
    bool success = true;
    try {
        Tensor<T>::allocate(ec, T1);
        try {
            Scheduler{ec}(T1() = -1.0)(LT1 = 42).execute();
            check_value(LT1, 42.0);
            for(const auto& lt : rest_lts) { check_value(lt, -1.0); }
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
        check_value(LT1, 42.0);
        for(const auto& lt : rest_lts) { check_value(lt, -1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 0. Exception: " << e << "\n";
        success = false;
    }

    try {
        success = true;
        Scheduler{ec}(T1() = -1.0)(LT1 = 4)(LT2 = 42)(LT1 += LT2).execute();
        check_value(LT1, 46.0);
        for(const auto& lt : rest_lts) { check_value(lt, -1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 1. Exception: " << e << "\n";
        success = false;
    }

    try {
        success = true;
        Scheduler{ec}(T1() = -1.0)(LT1 = 4)(LT2 = 42)(LT1 += 3 * LT2)
          .execute();
        check_value(T1, 130.0);
        for(const auto& lt : rest_lts) { check_value(lt, -1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 2. Exception: " << e << "\n";
        success = false;
    }

    try {
        success = true;
        Scheduler{ec}(T1() = -1.0)(T1() = 4)(T2() = 42)(T1() -= T2()).execute();
        check_value(T1, -38.0);
        for(const auto& lt : rest_lts) { check_value(lt, -1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 3. Exception: " << e << "\n";
        success = false;
    }

    try {
        success = true;
        Scheduler{ec}
          (T1() = -1.0)(T1() = 4)(T2() = 42)(T1() += -3.1 * T2())
          .execute();
        check_value(T1, -126.2);
        for(const auto& lt : rest_lts) { check_value(lt, -1.0); }
    } catch(std::string& e) {
        std::cerr << "AddOp. Test 4. Exception: " << e << "\n";
        success = false;
    }

    Tensor<T>::deallocate(T1, T2);
    return success;
}

//setop with T (call with tilesize 1 and 3)
template<typename T>
void test_setop_with_T(int tilesize) {
    //0-4 dimensional setops
    //0-4 dimensional setops

    bool failed;
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
    {
        Tensor<T> T1{TIS, TIS};
        REQUIRE(test_setop(ec, T1, T1(l1, l1)));
    }

    {
        Tensor<T> T1{TIS, TIS};
        REQUIRE(test_setop(ec, T1, T1(l1, l1), {T1(l1, l2), T1(l2, l1), T1(l2, l2)}));
    }
    REQUIRE(!failed);
    MemoryManagerGA::destroy_coll(mgr);
    delete ec;

}

//addop with T  (call with tilesize 1 and 3)
template<typename T>
void test_addop_with_T(int tilesize) {
    //0-4 dimensional addops

    //0-4 dimensional addops
}

TEST_CASE("setop with double") {
    test_setop_with_T<double>(1);
    test_setop_with_T<double>(3);
}

TEST_CASE("setop with float") {
    test_setop_with_T<float>(1);
    test_setop_with_T<float>(3);
}

TEST_CASE("addop with double") {
    test_addop_with_T<double>(1);
    test_addop_with_T<double>(3);
}

TEST_CASE("addop with float") {
    test_addop_with_T<float>(1);
    test_addop_with_T<float>(3);
}

TEST_CASE("Two-dimensional ops") {
    bool failed;
    ProcGroup pg{GA_MPI_Comm()};
    MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};
    using T              = double;

    IndexSpace IS{range(0, 10),
                  {{"nr1", {range(0, 5)}}, {"nr2", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, 1};
    TiledIndexLabel l1, l2, lall, lnone;
    std::tie(l1) = TIS.labels<1>("nr1");
    std::tie(l2) = TIS.labels<1>("nr2");
    std::tie(lall) = TIS.labels<1>("all");
    //@todo is there a "none" slice?
    //std::tie(lnone) = TIS.labels<1>("none");
    {
        Tensor<T> T1{TIS, TIS};
        REQUIRE(test_setop(ec, T1, T1(l1, l1)));
    }

    {
        Tensor<T> T1{TIS, TIS};
        REQUIRE(test_setop(ec, T1, T1(l1, l1), {T1(l1, l2), T1(l2, l1), T1(l2, l2)}));
    }
    REQUIRE(!failed);
    MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}

TEST_CASE("One-dimensional ops") {
    bool failed;
    ProcGroup pg{GA_MPI_Comm()};
    MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};
    using T              = double;

    IndexSpace IS{range(0, 10),
                  {{"nr1", {range(0, 5)}}, {"nr2", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, 1};
    TiledIndexLabel l1, l2, lall, lnone;
    std::tie(l1) = TIS.labels<1>("nr1");
    std::tie(l2) = TIS.labels<1>("nr2");
    std::tie(lall) = TIS.labels<1>("all");
    //@todo is there a "none" slice?
    //std::tie(lnone) = TIS.labels<1>("none");

    {
        Tensor<T> T1{TIS};
        REQUIRE(test_setop(ec, T1, T1()));
    }

    {
        Tensor<T> T1{TIS};
        //@todo Erdal: #else path fails. Can you check?
#if 1
        REQUIRE(test_setop(ec, T1, T1(l1)));
#else
        REQUIRE(test_setop(ec, T1, T1(l1), {T1(l2)}));
#endif
    }

    //@todo slice addop tests
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
        check_value(T1, 32.0);
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
        Scheduler{ec}
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
    MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}

TEST_CASE("Two-dimensional ops part I") {
    bool failed;
    ProcGroup pg{GA_MPI_Comm()};
    MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext* ec = new ExecutionContext{pg, &distribution, mgr};
    using T              = double;

    IndexSpace IS{range(0, 10),
                      {{"nr1", {range(0, 5)}}, {"nr2", {range(5, 10)}}}};
    TiledIndexSpace TIS{IS, 1};

    //setop
    {
        Tensor<T> T1{TIS, TIS};
        Tensor<T>::allocate(ec, T1);
        Scheduler{ec}(T1() = 42).execute();
        check_value(T1, 42.0);
        Tensor<T>::deallocate(T1);
    }

    //addop
    try {
        failed = false;
        Tensor<T> T1{TIS,TIS}, T2{TIS,TIS};
        Scheduler{ec}
        .allocate(T1, T2)
        (T2() = 42)
        (T1() = T2())
        .deallocate(T2)
        .execute();
        check_value(T1, 42.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS,TIS}, T2{TIS,TIS};
        Scheduler{ec}
        .allocate(T1, T2)
        (T1() = 4)
        (T2() = 42)
        (T1() += T2())
        .deallocate(T2)
        .execute();
        check_value(T1, 46.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS,TIS}, T2{TIS,TIS};
        Scheduler{ec}
        .allocate(T1, T2)
        (T1() = 4)
        (T2() = 42)
        (T1() += 3*T2())
        .deallocate(T2)
        .execute();
        check_value(T1, 130.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS,TIS}, T2{TIS,TIS};
        Scheduler{ec}
        .allocate(T1, T2)
        (T1() = 4)
        (T2() = 42)
        (T1() -= T2())
        .deallocate(T2)
        .execute();
        check_value(T1, -38.0);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS,TIS}, T2{TIS,TIS};
        Scheduler{ec}
        .allocate(T1, T2)
        (T1() = 4)
        (T2() = 42)
        (T1() += -3.1*T2())
        .deallocate(T2)
        .execute();
        check_value(T1, -126.2);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);


    //multop: 2,2,0
    try {
        failed = false;
        Tensor<T> T1{TIS,TIS}, T2{TIS,TIS}, T3{};
        Scheduler{ec}
        .allocate(T1, T2, T3)
        (T1() = 4)
        (T2() = 42)
        (T3() = 5)
        (T1() += -3.1*T2()*T3())
        .deallocate(T2, T3)
        .execute();
        check_value(T1, 4 -3.1*42*5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{TIS,TIS}, T2{TIS,TIS}, T3{};
        Scheduler{ec}
        .allocate(T1, T2, T3)
        (T1() = 4)
        (T2() = 42)
        (T3() = 5)
        (T1() -= -3.1*T2()*T3())
        .deallocate(T2, T3)
        .execute();
        check_value(T1, 4 +3.1*42*5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    try {
        failed = false;
        Tensor<T> T1{}, T2{TIS,TIS}, T3{TIS,TIS};
        Scheduler{ec}
        .allocate(T1, T2, T3)
        (T1() = 4)
        (T2() = 42)
        (T3() = 5)
        (T1() += -3.1*T2()*T3())
        .deallocate(T2, T3)
        .execute();
        check_value(T1, 4 -3.1*10*10*42*5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);

    //multop 2,1,1
    try {
        failed = false;
        Tensor<T> T1{TIS,TIS}, T2{TIS}, T3{TIS};
        TiledIndexLabel i, j;
        std::tie(i,j) = TIS.labels<2>("all");
        Scheduler{ec}
        .allocate(T1, T2, T3)
        (T1() = 4)
        (T2() = 42)
        (T3() = 5)
        (T1(i,j) += -3.1*T2(i)*T3(j))
        .deallocate(T2, T3)
        .execute();
        check_value(T1, 4 -3.1*42*5);
        Tensor<T>::deallocate(T1);
    } catch(std::string& e) {
        std::cerr << "Caught exception: " << e << "\n";
        failed = true;
    }
    REQUIRE(!failed);
}
