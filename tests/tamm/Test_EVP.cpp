#include <tamm/tamm.hpp>

#if defined(USE_SCALAPACK)

#include <blacspp/grid.hpp>
#include <scalapackpp/block_cyclic_matrix.hpp>
#include <scalapackpp/eigenvalue_problem/sevp.hpp>
#include <scalapackpp/pblas/gemm.hpp>

using namespace tamm;
using std::cout;
using std::endl;
using T = double;

struct ScalapackInfo {
  int64_t                                         npr{}, npc{}, scalapack_nranks{};
  std::unique_ptr<blacspp::Grid>                  blacs_grid;
  std::unique_ptr<scalapackpp::BlockCyclicDist2D> blockcyclic_dist;
  tamm::ProcGroup                                 pg;
  ExecutionContext                                ec;
};

std::tuple<int, int, int, int, int, int> sca_get_subgroup_info(ExecutionContext& gec,
                                                               const size_t      N) {
  auto nnodes = gec.nnodes();
  auto ppn    = gec.ppn();

  int hf_guessranks = std::ceil(0.9 * N);
  int hf_nnodes     = hf_guessranks / ppn;
  if(hf_guessranks % ppn > 0 || hf_nnodes == 0) hf_nnodes++;
  if(hf_nnodes > nnodes) hf_nnodes = nnodes;
  int hf_nranks = hf_nnodes * ppn;

  // Find nearest square
  int sca_nranks = std::ceil(N / 10);
  if(sca_nranks > hf_nranks) sca_nranks = hf_nranks;
  sca_nranks = std::pow(std::floor(std::sqrt(sca_nranks)), 2);
  if(sca_nranks == 0) sca_nranks = 1;
  int sca_nnodes = sca_nranks / ppn;
  if(sca_nranks % ppn > 0 || sca_nnodes == 0) sca_nnodes++;
  if(sca_nnodes > nnodes) sca_nnodes = nnodes;
  // if(sca_nnodes == 1) ppn = sca_nranks; // single node case

  return std::make_tuple(nnodes, hf_nnodes, ppn, hf_nranks, sca_nnodes, sca_nranks);
}

// TEST_CASE("Testing EVP")
void test_evp(size_t N, size_t mb) {
  ProcGroup        gpg = ProcGroup::create_world_coll();
  ExecutionContext gec{gpg, DistributionKind::nw, MemoryManagerKind::ga};

  auto [nnodes, hf_nnodes, ppn, hf_nranks, sca_nnodes, sca_nranks] = sca_get_subgroup_info(gec, N);

  auto rank = gec.pg().rank();
  gec.pg().barrier();

  if(rank == 0) {
    cout << "problem size = " << N << endl;
    cout << "block size   = " << mb << endl;
    cout << std::endl
         << "Number of nodes, mpi ranks per node provided: " << nnodes << ", " << gec.ppn() << endl;
    cout << "Number of nodes, mpi ranks per node used for calculation: " << hf_nnodes << ", " << ppn
         << endl;
    cout << "Number of nodes, mpi ranks per node, total ranks used for Scalapack: " << sca_nnodes
         << ", " << sca_nranks / sca_nnodes << ", " << sca_nranks << endl;
  }

  auto gcomm = gec.pg().comm();

  int ranks[hf_nranks];
  for(int i = 0; i < hf_nranks; i++) ranks[i] = i;
  MPI_Group wgroup;
  MPI_Comm_group(gcomm, &wgroup);
  MPI_Group hfgroup;
  MPI_Group_incl(wgroup, hf_nranks, ranks, &hfgroup);
  MPI_Comm hf_comm;
  MPI_Comm_create(gcomm, hfgroup, &hf_comm);

  int lranks[sca_nranks];
  for(int i = 0; i < sca_nranks; i++) lranks[i] = i;
  MPI_Group sca_group;
  MPI_Group_incl(wgroup, sca_nranks, lranks, &sca_group);
  MPI_Comm scacomm;
  MPI_Comm_create(gcomm, sca_group, &scacomm);

  if(rank < hf_nranks) {
    EXPECTS(hf_comm != MPI_COMM_NULL);
    ScalapackInfo scalapack_info;

    ProcGroup        pg = ProcGroup::create_coll(hf_comm);
    ExecutionContext ec{pg, DistributionKind::dense, MemoryManagerKind::ga};

    TiledIndexSpace AO_sca{IndexSpace{range(N)}, static_cast<Tile>(mb)};

    int world_size = sca_nranks; // ec.pg().size().value();

    // Default to square(ish) grid
    int64_t npr = std::sqrt(world_size);
    int64_t npc = world_size / npr;
    while(npr * npc != world_size) {
      npr--;
      npc = world_size / npr;
    }

    int scalapack_nranks = world_size;
    scalapack_info.npr   = npr;
    scalapack_info.npc   = npc;

    assert(world_size >= scalapack_nranks);

    std::vector<int64_t> scalapack_ranks(scalapack_nranks);
    std::iota(scalapack_ranks.begin(), scalapack_ranks.end(), 0);
    scalapack_info.scalapack_nranks = scalapack_nranks;

    if(rank == 0) {
      std::cout << "scalapack_nranks = " << scalapack_nranks << std::endl;
      std::cout << "scalapack_np_row = " << scalapack_info.npr << std::endl;
      std::cout << "scalapack_np_col = " << scalapack_info.npc << std::endl;
    }

    if(scacomm != MPI_COMM_NULL) {
      auto blacs_setup_st = std::chrono::high_resolution_clock::now();

      scalapack_info.pg = ProcGroup::create_coll(scacomm);
      scalapack_info.ec =
        ExecutionContext{scalapack_info.pg, DistributionKind::dense, MemoryManagerKind::ga};
      scalapack_info.blacs_grid = std::make_unique<blacspp::Grid>(
        scalapack_info.pg.comm(), scalapack_info.npr, scalapack_info.npc, scalapack_ranks.data(),
        scalapack_info.npr);
      scalapack_info.blockcyclic_dist =
        std::make_unique<scalapackpp::BlockCyclicDist2D>(*scalapack_info.blacs_grid, mb, mb, 0, 0);

      auto blacs_setup_en = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> blacs_time = blacs_setup_en - blacs_setup_st;

      if(rank == 0) {
        std::cout << std::fixed << std::setprecision(2) << std::endl
                  << "Time for BLACS setup: " << blacs_time.count() << " secs" << std::endl;
      }

      blacs_setup_st                                   = std::chrono::high_resolution_clock::now();
      blacspp::Grid*                  blacs_grid       = scalapack_info.blacs_grid.get();
      scalapackpp::BlockCyclicDist2D* blockcyclic_dist = scalapack_info.blockcyclic_dist.get();

      auto desc_lambda = [&](const int64_t M, const int64_t N) {
        auto [M_loc, N_loc] = (*blockcyclic_dist).get_local_dims(M, N);
        return (*blockcyclic_dist).descinit_noerror(M, N, M_loc);
      };

      const auto& grid   = *blacs_grid;
      const auto  mb     = blockcyclic_dist->mb();
      const auto  Northo = N;

      Tensor<T> F_BC{AO_sca, AO_sca};
      Tensor<T> X_alpha{AO_sca, AO_sca};
      F_BC.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
      X_alpha.set_block_cyclic({scalapack_info.npr, scalapack_info.npc});
      Tensor<T>::allocate(&scalapack_info.ec, X_alpha, F_BC);
      // Scheduler{scalapack_info.ec}(F_BC() = X_alpha()* X_alpha()).execute();

      if(grid.ipr() >= 0 and grid.ipc() >= 0) {
        // TODO: Optimize intermediates here
        scalapackpp::BlockCyclicMatrix<double> Fp_sca(grid, Northo, Northo, mb, mb),
          Ca_sca(grid, Northo, Northo, mb, mb), TMP1_sca(grid, N, Northo, mb, mb),
          TMP2_sca(grid, Northo, N, mb, mb);

        auto desc_Fa = desc_lambda(N, N);
        auto desc_Xa = desc_lambda(Northo, N);

        //   tamm::to_block_cyclic_tensor(ttensors.F_alpha, ttensors.F_BC);
        scalapack_info.pg.barrier();

        blacs_setup_en = std::chrono::high_resolution_clock::now();
        blacs_time     = blacs_setup_en - blacs_setup_st;
        if(scalapack_info.pg.rank() == 0)
          std::cout << std::fixed << std::setprecision(2) << std::endl
                    << "Diagonalize setup time: " << blacs_time.count() << " secs" << std::endl;

        blacs_setup_st = std::chrono::high_resolution_clock::now();

        auto Fa_tamm_lptr = F_BC.access_local_buf();
        auto Xa_tamm_lptr = X_alpha.access_local_buf();

        // Compute TMP = F * X -> F * X**T (b/c row-major)
        // scalapackpp::pgemm( scalapackpp::Op::NoTrans, scalapackpp::Op::Trans,
        // 1., Fa_sca, Xa_sca, 0., TMP1_sca );
        scalapackpp::pgemm(scalapackpp::Op::NoTrans, scalapackpp::Op::Trans, TMP1_sca.m(),
                           TMP1_sca.n(), desc_Fa[3], 1., Fa_tamm_lptr, 1, 1, desc_Fa, Xa_tamm_lptr,
                           1, 1, desc_Xa, 0., TMP1_sca.data(), 1, 1, TMP1_sca.desc());

        // Compute Fp = X**T * TMP -> X * TMP (b/c row-major)
        // scalapackpp::pgemm( scalapackpp::Op::NoTrans, scalapackpp::Op::NoTrans,
        // 1., Xa_sca, TMP1_sca, 0., Fp_sca );
        scalapackpp::pgemm(scalapackpp::Op::NoTrans, scalapackpp::Op::NoTrans, Fp_sca.m(),
                           Fp_sca.n(), desc_Xa[3], 1., Xa_tamm_lptr, 1, 1, desc_Xa, TMP1_sca.data(),
                           1, 1, TMP1_sca.desc(), 0., Fp_sca.data(), 1, 1, Fp_sca.desc());

        // Solve EVP
        std::vector<T> eps_a(Northo);
        // scalapackpp::hereigd( scalapackpp::Job::Vec, scalapackpp::Uplo::Lower,
        //                       Fp_sca, eps_a.data(), Ca_sca );
        auto info = scalapackpp::hereig(scalapackpp::Job::Vec, scalapackpp::Uplo::Lower, Fp_sca.m(),
                                        Fp_sca.data(), 1, 1, Fp_sca.desc(), eps_a.data(),
                                        Ca_sca.data(), 1, 1, Ca_sca.desc());

        // Backtransform TMP = X * Ca -> TMP**T = Ca**T * X
        // scalapackpp::pgemm( scalapackpp::Op::Trans, scalapackpp::Op::NoTrans,
        //                     1., Ca_sca, Xa_sca, 0., TMP2_sca );
        scalapackpp::pgemm(scalapackpp::Op::Trans, scalapackpp::Op::NoTrans, TMP2_sca.m(),
                           TMP2_sca.n(), Ca_sca.m(), 1., Ca_sca.data(), 1, 1, Ca_sca.desc(),
                           Xa_tamm_lptr, 1, 1, desc_Xa, 0., TMP2_sca.data(), 1, 1, TMP2_sca.desc());

        blacs_setup_en = std::chrono::high_resolution_clock::now();

        blacs_time = blacs_setup_en - blacs_setup_st;

        if(scalapack_info.pg.rank() == 0)
          std::cout << std::fixed << std::setprecision(2) << std::endl
                    << "Diagonalize time: " << blacs_time.count() << " secs" << std::endl;
      }

      // Gather results
      //   if(rank == 0) C_alpha.resize(N, Northo);
      //   TMP2_sca.gather_from(Northo, N, C_alpha.data(), Northo, 0, 0);
      Tensor<T>::deallocate(X_alpha, F_BC);

    } // scalapack comm

  } // hf nranks

} // test_svp
#endif

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  size_t N = 100, mb = 32;
#if defined(USE_SCALAPACK)
  if(argc >= 2) N = std::atoi(argv[1]);
  if(argc == 3) mb = std::atoi(argv[2]);
  test_evp(N, mb);
#endif

  tamm::finalize();
  return 0;
}
