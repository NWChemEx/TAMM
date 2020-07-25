#include "ga-mpi.h"
#include "ga.h"
#include "macdecls.h"
#include "mpi.h"
#include <tamm/tamm.hpp>

using namespace tamm;
using std::cout;
using std::endl;
using T = double;

// TEST_CASE("/* Testing process groups */") 
void test_pg(int dim, int nproc) {

    ProcGroup gpg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext gec{gpg, DistributionKind::dense, MemoryManagerKind::ga};

    auto rank = gec.pg().rank();

    auto subranks=nproc;
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

            ProcGroup pg = ProcGroup::create_coll(subcomm);
            ExecutionContext ec{pg, DistributionKind::dense,
                                MemoryManagerKind::ga};

            TiledIndexSpace tis1{IndexSpace{range(dim)}, 40};

            auto [i, j, k] = tis1.labels<3>("all");

            Tensor<T> A{i, k};
            Tensor<T> B{k, j};
            Tensor<T> C{i, j};

            Scheduler{ec}.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).deallocate(A,B,C).execute();

            ec.flush_and_sync();
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
}

// TEST_CASE("/* Test case for replicated C */") {

//     ProcGroup gpg{GA_MPI_Comm()};
//     auto gmgr = MemoryManagerGA::create_coll(gpg);
//     Distribution_NW gdistribution;
//     RuntimeEngine gre;
//     ExecutionContext gec{gpg, &gdistribution, gmgr, &gre};

//     TiledIndexSpace tis1{IndexSpace{range(20)}, 2};

//     auto [i, j, k] = tis1.labels<3>("all");

//     auto rank = gec.pg().rank();
//     Tensor<T> A{i, k};
//     Tensor<T> B{k, j};
//     Scheduler gsch{gec};
//     gsch.allocate(A, B).execute();

//     {

//         ProcGroup pg{MPI_COMM_SELF};
//         auto mgr = MemoryManagerLocal::create_coll(pg);
//         Distribution_NW distribution;
//         RuntimeEngine re;
//         ExecutionContext ec{pg, &distribution, mgr, &re};

//         Tensor<T> C{i, j};

//         Scheduler{ec}.allocate(C).execute();
        
//         gsch
//         (A() = 21.0)(B() = 2.0)(C() = A()*B()).execute();

//         Scheduler{ec}.deallocate(C).execute();

//         ec.flush_and_sync();
//         MemoryManagerLocal::destroy_coll(mgr);
//     }

//     gec.pg().barrier();

//     gsch.deallocate(A, B).execute();

//     gec.flush_and_sync();
//     MemoryManagerGA::destroy_coll(gmgr);
    
// }

// TODO: Add test for replicated A/B on sub-comm, ie A/B are 
// shared across ranks in sub-comm - use MemoryManagerGA

// TEST_CASE("/* Test case for replicated A/B */")
void test_replicate_AB(int dim) {

    ProcGroup gpg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext gec{gpg, DistributionKind::dense, MemoryManagerKind::ga};

    TiledIndexSpace tis1{IndexSpace{range(dim)}, 40};

    auto [i, j, k] = tis1.labels<3>("all");

    auto rank = gec.pg().rank();
    Tensor<T> A{tis1, tis1};
    Tensor<T> B{tis1, tis1};
    Tensor<T> C{tis1, tis1};

    if(gec.pg().rank()==0) cout << "N=" << dim << endl;
    Scheduler gsch{gec};
    gsch.allocate(A, C).execute();

    { // B is replicated

        ProcGroup pg = ProcGroup::create_coll(MPI_COMM_SELF);
        ExecutionContext ec{pg, DistributionKind::dense, MemoryManagerKind::ga};

        Scheduler{ec}.allocate(B).execute();
        
        gsch
        (A() = 21.0)(B() = 2.0)(C(i,j) = A(i,k)*B(k,j)).execute();

        Scheduler{ec}.deallocate(B).execute();

        ec.flush_and_sync();
    }

    gec.pg().barrier();

    gsch.deallocate(A, C).execute();

    gec.flush_and_sync();
    
}


int main(int argc, char* argv[]) {
    
    tamm::initialize(argc,argv);

    auto dim = 20;
    int nproc = 20;
    if(argc == 2) dim = std::atoi(argv[1]);
    if(argc == 3) nproc = std::atoi(argv[2]);
    test_pg(dim,nproc);
    //test_replicate_AB(dim);
    
    tamm::finalize();
    return 0;
}
