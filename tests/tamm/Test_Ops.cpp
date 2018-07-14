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
    for(auto i=0;i<size;i++)
      buf[i]=cx++;
    T1.put(IndexVector{1,0,1}, span<T>(buf,size));

    T* gbuf = new T[size];
    T1.get(IndexVector{1,0,1}, span<T>(gbuf,size));
    for(auto i=0;i<size;i++)
        EXPECTS(gbuf[i]==buf[i]);
        
    Tensor<T>::deallocate(T1);

    Tensor<T> d_evl{N,N};
    Tensor<T> xt{N,N};
    //@todo Set EVL to have local distribution (one copy in each MPI rank)
    Tensor<T>::allocate(ec, d_evl,xt);
  
    Scheduler{ec}
        (d_evl("n1","n2") = 2.2)
        (xt("n1","n2") = 2.0*d_evl("n1","n2"))
        .execute();

    for (auto it: d_evl.loop_nest())
    {
        auto size = d_evl.block_size(it);
        T* buf = new T[size];
        d_evl.get(it,span<T>(buf,size));
        for (auto i = 0; i < size;i++)
         EXPECTS(buf[i]==2.2);
    }

    for (auto it: xt.loop_nest())
    {
        auto size = xt.block_size(it);
        T* buf = new T[size];
        xt.get(it,span<T>(buf,size));
        for (auto i = 0; i < size;i++)
         EXPECTS(buf[i]==4.4);
    }

    Tensor<T>::deallocate(d_evl,xt);

}

int main( int argc, char* argv[] )
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

TEST_CASE("CCSD Driver") {
    // Construction of tiled index space MO from sketch
    IndexSpace MO_IS{range(0, 200),
                     {{"occ", {range(0, 100)}}, {"virt", {range(100, 200)}}}};
    TiledIndexSpace MO{MO_IS, 10};

    const TiledIndexSpace& N = MO("all");

    CHECK_NOTHROW(test_ops<double>(MO));
}
