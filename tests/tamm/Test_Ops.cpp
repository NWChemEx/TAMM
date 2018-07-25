//#define CATCH_CONFIG_MAIN
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
        std::cerr<<__FUNCTION__<<" "<<__LINE__<<"\n";
        TAMM_SIZE size = t.block_size(it);
        T* buf = new T[size];
        t.get(it,span<T>(buf,size));
        for (TAMM_SIZE i = 0; i < size;i++) {
            std::cerr<<"Calling expects\n";
          REQUIRE(buf[i]==val);
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
        Scheduler{ec}(T2() = 42)(T1() = T2()).execute();
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
          .allocate(T1, T2)(T1()=3)(T2() = 42)(T1() += 2.5*T2())
          .deallocate(T2)
          .execute();
        check_value(T1, 49.5);
        Tensor<T>::deallocate(T1);
    }

    {
        // Tensor<T> T1{}, T2{};
        // Scheduler{ec}
        //   .allocate(T1, T2)(T1()=3)(T2() = 42)(T1() -= T2())
        //   .deallocate(T2)
        //   .execute();
        // check_value(T1, 39.0);
        // Tensor<T>::deallocate(T1);
    }

    {
        // Tensor<T> T1{}, T2{};
        // Scheduler{ec}
        //   .allocate(T1, T2)(T1()=3)(T2() = 42)(T1() -= 4*T2())
        //   .deallocate(T2)
        //   .execute();
        // check_value(T1, 30.0);
        // Tensor<T>::deallocate(T1);
    }

    // {
    //     Tensor<T> T1{};
    //     Scheduler{ec}.allocate(T1)(T1() = 42).execute();
    //     check_value(T1, 42.0);
    //     Tensor<T>::deallocate(T1);
    // }

    // {
    //     Tensor<T> T1{};
    //     Scheduler{ec}
    //         .allocate(T1)
    //         (T1() = 42)
    //         .execute();
    //     check_value(T1, 42.0);
    //     Tensor<T>::deallocate(T1);
    // }

    MemoryManagerGA::destroy_coll(mgr);
    delete ec;
}
