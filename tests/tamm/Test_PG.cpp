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

TEST_CASE("/* Testing process groups */") {

    ProcGroup gpg{GA_MPI_Comm()};
    auto gmgr = MemoryManagerGA::create_coll(gpg);
    Distribution_NW gdistribution;
    RuntimeEngine gre;
    ExecutionContext gec{gpg, &gdistribution, gmgr, &gre};

    auto rank = gec.pg().rank();

    auto subranks=30;
    int ranks[subranks];
    for (int i = 0; i < subranks; i++) ranks[i] = i;

    if(subranks < GA_Nnodes()) {
        auto world_comm = gec.pg().comm();
        MPI_Group world_group;
        MPI_Comm_group(world_comm,&world_group);
        MPI_Group subgroup;
        MPI_Group_incl(world_group,subranks,ranks,&subgroup);
        MPI_Comm subcomm;
        MPI_Comm_create(world_comm,subgroup,&subcomm);

        if (rank < subranks)  {
            int hrank;
            EXPECTS(subcomm != MPI_COMM_NULL);
            MPI_Comm_rank(subcomm,&hrank);
            EXPECTS(rank==hrank);

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
        // MPI_Group_free(&world_group);
        // MPI_Group_free(&subgroup);
        // MPI_Comm_free(&world_comm);
    }

    gec.pg().barrier();

    TiledIndexSpace AO{IndexSpace{range(20)},2};
    Tensor<double> T0{AO, AO};
    T0.allocate(&gec);
    Tensor<T>::deallocate(T0);
    gec.flush_and_sync();
    MemoryManagerGA::destroy_coll(gmgr);
    
}

TEST_CASE("/* Test case for replicated tensors */") {

    ProcGroup gpg{GA_MPI_Comm()};
    auto gmgr = MemoryManagerGA::create_coll(gpg);
    Distribution_NW gdistribution;
    RuntimeEngine gre;
    ExecutionContext gec{gpg, &gdistribution, gmgr, &gre};

    TiledIndexSpace tis1{IndexSpace{range(20)}, 2};

    auto [i, j, k] = tis1.labels<3>("all");

    auto rank = gec.pg().rank();
    Tensor<T> A{i, k};
    Tensor<T> B{k, j};
    Scheduler gsch{gec};
    gsch.allocate(A, B).execute();

    {

        ProcGroup pg{MPI_COMM_SELF};
        auto mgr = MemoryManagerLocal::create_coll(pg);
        Distribution_NW distribution;
        RuntimeEngine re;
        ExecutionContext ec{pg, &distribution, mgr, &re};

        Tensor<T> C{i, j};

        Scheduler{ec}.allocate(C).execute();
        
        gsch
        (A() = 21.0)(B() = 2.0)(C() = A()*B()).execute();

        Scheduler{ec}.deallocate(C).execute();

        ec.flush_and_sync();
        MemoryManagerLocal::destroy_coll(mgr);
    }

    gec.pg().barrier();

    gsch.deallocate(A, B).execute();

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