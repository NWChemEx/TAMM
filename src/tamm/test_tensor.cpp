
#include "ga.h"
#include "mpi.h"
#include "macdecls.h"
#include "ga-mpi.h"
#include "tamm/tamm.hpp"

using namespace tamm;

template<typename T>
void test_tensor(const TiledIndexSpace& MO) {
    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    const TiledIndexSpace& N = MO("all");

    Tensor<T> d_t1{V, O};
    Tensor<T> d_t2{V, V, O, O};

    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    
    ExecutionContext *ec = new ExecutionContext{pg,&distribution,mgr};

    TiledIndexSpace UnitTiledMO{MO.index_space(), 1};
    Tensor<T> d_evl{N,N,N};
    //@todo Set EVL to have local distribution (one copy in each MPI rank)
    Tensor<T>::allocate(ec, d_evl);

    T cx=42;
    size_t size = 1000;
    T* buf = new T[size];
    for(auto i=0;i<size;i++)
      buf[i]=cx++;
    d_evl.put(IndexVector{1,0,1}, span<T>(buf,size));

    T* gbuf = new T[size];
    d_evl.get(IndexVector{1,0,1}, span<T>(gbuf,size));
    for(auto i=0;i<size;i++)
        EXPECTS(gbuf[i]==buf[i]);
        
    Tensor<T>::deallocate(d_evl);
}

int main( int argc, char* argv[] )
{
    MPI_Init(&argc,&argv);
    GA_Initialize();
    MA_init(MT_DBL, 8000000, 20000000);
    
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    // Construction of tiled index space MO from sketch
    IndexSpace MO_IS{range(0, 20),
                     {{"occ", {range(0, 10)}}, {"virt", {range(10, 20)}}}};
    TiledIndexSpace MO{MO_IS, 10};

    const TiledIndexSpace& N = MO("all");

    test_tensor<double>(MO);
    GA_Terminate();
    MPI_Finalize();
}
