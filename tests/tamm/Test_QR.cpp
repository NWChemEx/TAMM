// #define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_RUNNER
#include <catch/catch.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include "ga-mpi.h"
#include "ga.h"
#include "macdecls.h"
#include "mpi.h"
#include <tamm/tamm.hpp>

using namespace tamm;
using std::cout;
using std::endl;
using T = std::complex<double>;

TEST_CASE("/* Test case for QR */") {

    using Complex2DMatrix=Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    RuntimeEngine re;
    ExecutionContext ec{pg, &distribution, mgr, &re};
    auto rank = ec.pg().rank();

    const TAMM_SIZE N = 10; //N is atmost 1000 for now
    int count = (N/20 >= 10) ? N/20 : 10; 

    TiledIndexSpace NIS{range(0, N), 40};
    //  auto [a,b,c] = N.labels<3>("all");

    //some factor by which count increases, can go upto 50 for N=1000
    const int factor = 1; 

    //QR is performed within a loop where count keeps increasing every iteration
    while(true) {

        TiledIndexSpace AUX{IndexSpace{range(count)}};
        // auto [z] = AUX.labels<1>("all");

        Tensor<T> A{NIS,AUX};
        Tensor<T> B{NIS,NIS,NIS,AUX};

        Scheduler sch{ec};
        sch.allocate(A, B).execute();
        // Initialize A,B
        
        // copy tamm tensors A,B to AB_combined. To keep the unit test simple,
        // fill AB_Combined with random values
        Complex2DMatrix AB_combined=Complex2DMatrix::Random(N + N*N*N, count);
        sch.deallocate(A, B).execute();

        
        Eigen::FullPivHouseholderQR<Complex2DMatrix> qr(AB_combined);
        auto qr_rank = qr.rank();
        Complex2DMatrix thinQ(Complex2DMatrix::Identity
            (AB_combined.rows(),qr_rank));
        Complex2DMatrix AB_Q = qr.matrixQ() * thinQ;

        //We combined A,B to call QR - Split resulting AB_Q 
        //into A_Q1(N,count) and B_Q2(N,N,N,count)
        //and copy back to tamm A_Q1 and B_Q2. The thinning  
        //should actually be done in tamm after the copy.

        //do_work(A_Q1,A_Q2)

        count += factor;

        break; //break when converged 

    }

    ec.flush_and_sync();
    MemoryManagerGA::destroy_coll(mgr);
    
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