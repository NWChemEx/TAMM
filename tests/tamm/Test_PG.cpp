// #define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_RUNNER
#include <catch/catch.hpp>

#include "ga-mpi.h"
#include "ga.h"
#include "macdecls.h"
#include "mpi.h"
#include <tamm/tamm.hpp>

using namespace tamm;
using std::cout;
using std::endl;
using T = double;

TEST_CASE("/* Test case for getting ExecutionContext from a Tensor */") {

    ProcGroup gpg{GA_MPI_Comm()};
    auto gmgr = MemoryManagerGA::create_coll(gpg);
    Distribution_NW gdistribution;
    RuntimeEngine gre;
    ExecutionContext gec{gpg, &gdistribution, gmgr, &gre};

    auto rank = gec.pg().rank();

    auto subranks=30;
     int ranks[subranks];
     for (int i = 0; i < subranks; i++) ranks[i] = i;

    auto world_comm = gec.pg().comm();
    MPI_Group world_group;
    MPI_Comm_group(world_comm,&world_group);
    MPI_Group subgroup;
    MPI_Group_incl(world_group,subranks,ranks,&subgroup);
    MPI_Comm subcomm;
    MPI_Comm_create(world_comm,subgroup,&subcomm);


    if (rank < subranks) {

        int hrank;
        EXPECTS(subcomm != MPI_COMM_NULL);
        MPI_Comm_rank(subcomm,&hrank);
        EXPECTS(rank==hrank);
        // cout << "rank,hrank = " << rank << "," << hrank << endl;

        ProcGroup pg{subcomm};
        auto mgr = MemoryManagerGA::create_coll(pg);
        Distribution_NW distribution;
        RuntimeEngine re;
        ExecutionContext ec{pg, &distribution, mgr, &re};

        TiledIndexSpace tis1{IndexSpace{range(20)}, 2};

        auto [i, j, k] = tis1.labels<3>("all");

        Tensor<T> A{i, k};
        Tensor<T> B{k, j};
        Tensor<T> C{i, j};

        Scheduler{ec}.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).deallocate(A,B,C).execute();

        ec.flush_and_sync();
        MemoryManagerGA::destroy_coll(mgr);
    }

    gec.pg().barrier();

    if(rank==0) cout << "debug1" << endl;

    TiledIndexSpace AO{IndexSpace{range(20)},2};
    Tensor<double> T0{AO, AO};
    T0.allocate(&gec);
    if(rank==0) cout << "debug2" << endl;
      // MPI_Group_free(&world_group);
      // MPI_Group_free(&subgroup);
      // MPI_Comm_free(&world_comm);

    Tensor<T>::deallocate(T0);
    gec.flush_and_sync();
    MemoryManagerGA::destroy_coll(gmgr);
    
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