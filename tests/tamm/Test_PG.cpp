#include <tamm/tamm.hpp>

using namespace tamm;
using std::cout;
using std::endl;
using T = double;

// TEST_CASE("/* Testing process groups */")
void test_pg(int dim, int nproc) {
  ProcGroup        gpg = ProcGroup::create_world_coll();
  ExecutionContext gec{gpg, DistributionKind::nw, MemoryManagerKind::ga};

  auto rank = gec.pg().rank();

  auto      world_comm = gec.pg().comm();
  MPI_Group world_group;
  MPI_Comm_group(world_comm, &world_group);

  auto ppn = gec.ppn();
  if(rank == 0) std::cout << "ppn=" << ppn << std::endl;

#if defined(USE_UPCXX)
  upcxx::team* subcomm =
    new upcxx::team(upcxx::world().split(rank < ppn ? 0 : upcxx::team::color_none, 0));
#else
  int lranks[ppn];
  for(int i = 0; i < ppn; i++) lranks[i] = i;
  MPI_Group lgroup;
  MPI_Comm_group(world_comm, &lgroup);
  MPI_Group hf_lgroup;
  MPI_Group_incl(lgroup, ppn, lranks, &hf_lgroup);
  MPI_Comm subcomm;
  MPI_Comm_create(world_comm, hf_lgroup, &subcomm);
  MPI_Group_free(&lgroup);
  MPI_Group_free(&hf_lgroup);
#endif

  TiledIndexSpace AO{IndexSpace{range(5)}, 5};
  Tensor<double>  tens1{{AO, AO}};

  if(rank < ppn) {
    ProcGroup        pg_m = ProcGroup::create_coll(subcomm);
    ExecutionContext ec_m{pg_m, DistributionKind::nw, MemoryManagerKind::ga};
    Scheduler        sch{ec_m};

    sch.allocate(tens1)(tens1() = 4.0).execute();

    if(ec_m.pg().rank() == 0) {
      double* tptr = tens1.access_local_buf();
      auto    ts   = tens1.local_buf_size();
      if(ec_m.pg().rank() == 0) std::cout << "ts = " << ts << "\n";
      // for (int i=0;i<ts;i++) cout << tptr[i] << ",";
      // cout << "\n";
    }

    Tensor<T>::deallocate(tens1);
  }
  // gec.flush_and_sync();
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
  ProcGroup gpg = ProcGroup::create_world_coll();

  ExecutionContext gec{gpg, DistributionKind::dense, MemoryManagerKind::ga};

  TiledIndexSpace tis1{IndexSpace{range(dim)}, 40};

  auto [i, j, k] = tis1.labels<3>("all");

  auto      rank = gec.pg().rank();
  Tensor<T> A{tis1, tis1};
  Tensor<T> B{tis1, tis1};
  Tensor<T> C{tis1, tis1};

  if(gec.pg().rank() == 0) cout << "N=" << dim << endl;
  Scheduler gsch{gec};
  gsch.allocate(A, C).execute();

  { // B is replicated
#if defined(USE_UPCXX)
    upcxx::team self_team = upcxx::world().split(upcxx::rank_me(), 0);
    ProcGroup   pg        = ProcGroup::create_coll(self_team);
#else
    ProcGroup pg = ProcGroup::create_coll(MPI_COMM_SELF);
#endif
    ExecutionContext ec{pg, DistributionKind::dense, MemoryManagerKind::ga};

    Scheduler{ec}.allocate(B).execute();

    gsch(A() = 21.0)(B() = 2.0)(C(i, j) = A(i, k) * B(k, j)).execute();

    Scheduler{ec}.deallocate(B).execute();

    ec.flush_and_sync();
#if defined(USE_UPCXX)
    self_team.destroy();
#endif
  }

  gec.pg().barrier();

  gsch.deallocate(A, C).execute();

  gec.flush_and_sync();
}

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  auto dim   = 20;
  int  nproc = 20;
  if(argc == 2) dim = std::atoi(argv[1]);
  if(argc == 3) nproc = std::atoi(argv[2]);
  test_pg(dim, nproc);
  // test_replicate_AB(dim);

  tamm::finalize();
  return 0;
}
