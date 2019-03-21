// #define CATCH_CONFIG_RUNNER

#include "eomccsd.hpp"

template<typename T>
void eom_gradients(ExecutionContext& ec, const TiledIndexSpace& MO,
                   Tensor<T>& f1, Tensor<T>& v2,
                   Tensor<T>& t1, Tensor<T>& t2,
                   std::vector<Tensor<T>>& xc1, std::vector<Tensor<T>>& xc2,
                   std::vector<Tensor<T>>& yc1, std::vector<Tensor<T>>& yc2,
                   std::vector<T> omegar, std::vector<T> omegal, 
                   int targetroot, long int total_orbitals) {

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  const TiledIndexSpace& N = MO("all");

  auto [h1, h2] = MO.labels<2>("occ");
  auto [p3, p4] = MO.labels<2>("virt");

  auto rank = ec.pg().rank();
  Scheduler sch{ec};
  std::cout.precision(15);

  if(rank==0) std::cout << "\nTARGET ROOT FOR GRADIENTS: " << targetroot << std::endl;

  Tensor<T> d_r1{};
  Tensor<T>::allocate(&ec, d_r1);

  print_tensor(xc1.at(targetroot));
  print_tensor(yc1.at(targetroot));

  sch(d_r1()  = xc1.at(targetroot)(p3,h1) * yc1.at(targetroot)(h1,p3))
     (d_r1() += xc2.at(targetroot)(p3,p4,h1,h2) * yc2.at(targetroot)(h1,h2,p3,p4)).execute();

  if(rank==0) std::cout << "\nBIORTHOGONALITY TEST: " << get_scalar(d_r1) << std::endl;

  Tensor<T>::deallocate(d_r1);
}

void ccsd_driver();
std::string filename; //bad, but no choice

int main( int argc, char* argv[] )
{
    if(argc<2){
        std::cout << "Please provide an input file!\n";
        return 1;
    }

    filename = std::string(argv[1]);
    std::ifstream testinput(filename); 
    if(!testinput){
        std::cout << "Input file provided [" << filename << "] does not exist!\n";
        return 1;
    }

    MPI_Init(&argc,&argv);
    GA_Initialize();
    MA_init(MT_DBL, 8000000, 20000000);

    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    ccsd_driver();

    GA_Terminate();
    MPI_Finalize();

    return 0;
}


void ccsd_driver() {

    // std::cout << "Input file provided = " << filename << std::endl;

    using T = double;

    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    RuntimeEngine re;
    ExecutionContext ec{pg, &distribution, mgr, &re};
    auto rank = ec.pg().rank();

    //TODO: read from input file, assume no freezing for now
    TAMM_SIZE freeze_core    = 0;
    TAMM_SIZE freeze_virtual = 0;

    auto [options_map, ov_alpha, nao, hf_energy, shells, shell_tile_map, C_AO, F_AO, AO_opt, AO_tis] 
                    = hartree_fock_driver<T>(ec,filename);

    CCSDOptions ccsd_options = options_map.ccsd_options;
    if(rank == 0) ccsd_options.print();

    int maxiter    = ccsd_options.maxiter;
    double thresh  = ccsd_options.threshold;
    double zshiftl = 0.0;
    size_t ndiis   = 5;

    auto [MO,total_orbitals] = setupMOIS(ccsd_options.tilesize,
                  nao,ov_alpha,freeze_core,freeze_virtual);

    //deallocates F_AO, C_AO
    auto [cholVpr,d_f1,chol_count, max_cvecs] = cd_svd_driver<T>
                        (options_map, ec, MO, AO_opt, ov_alpha, nao, freeze_core,
                                freeze_virtual, C_AO, F_AO, shells, shell_tile_map);


  auto [p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s] 
                        = setupTensors(ec,MO,d_f1,ndiis);

  Tensor<T> d_v2 = setupV2<T>(ec,MO,cholVpr,chol_count, total_orbitals, ov_alpha, nao - ov_alpha);
  Tensor<T>::deallocate(cholVpr);

  auto cc_t1 = std::chrono::high_resolution_clock::now();
  auto [residual, energy] = ccsd_spin_driver<T>(ec, MO, d_t1, d_t2, d_f1, d_v2, 
                              d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, p_evl_sorted, 
                              maxiter, thresh, zshiftl, ndiis, 2 * ov_alpha);

  ccsd_stats(ec, hf_energy,residual,energy,thresh);

  auto cc_t2 = std::chrono::high_resolution_clock::now();

  double ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
  if(rank == 0) std::cout << "\nTime taken for CCSD: " << ccsd_time << " secs\n";

  free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);

//EOMCCSD Variables
    int nroots           = ccsd_options.eom_nroots;
    int maxeomiter       = ccsd_options.maxiter;
//    int eomsolver        = 1; //INDICATES WHICH SOLVER TO USE. (LATER IMPLEMENTATION)
    double eomthresh     = ccsd_options.eom_threshold;
//    double x2guessthresh = 0.6; //THRESHOLD FOR X2 INITIAL GUESS (LATER IMPLEMENTATION)
    size_t microeomiter  = ccsd_options.eom_microiter; //Number of iterations in a microcycle


//Right EOMCCSD Routine:
  cc_t1 = std::chrono::high_resolution_clock::now();

  auto [xc1,xc2,omegar] = right_eomccsd_driver<T>(ec, MO, d_t1, d_t2, d_f1, d_v2, p_evl_sorted,
                      nroots, maxeomiter, eomthresh, microeomiter,
                      total_orbitals, 2 * ov_alpha);

  cc_t2 = std::chrono::high_resolution_clock::now();

  ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<T>>((cc_t2 - cc_t1)).count();
  if(rank==0) std::cout << "\nTime taken for EOMCCSD: " << ccsd_time << " secs\n";

  //Left EOMCCSD Routine:
  cc_t1 = std::chrono::high_resolution_clock::now();

  auto [yc1,yc2,omegal] = left_eomccsd_driver<T>(ec, MO, d_t1, d_t2, d_f1, d_v2, p_evl_sorted,
                      nroots, maxeomiter, eomthresh, microeomiter,
                      total_orbitals, 2 * ov_alpha);

  cc_t2 = std::chrono::high_resolution_clock::now();

  ccsd_time =
    std::chrono::duration_cast<std::chrono::duration<T>>((cc_t2 - cc_t1)).count();
  if(rank==0) std::cout << "\nTime taken for Left-Eigenstate EOMCCSD: " << ccsd_time << " secs\n";

  free_tensors(d_r1, d_r2);

  // *************** BEGIN GRADIENTS CODE ************************* //
      int targetroot = 1;

    eom_gradients<T>(ec, MO, d_f1, d_v2, d_t1, d_t2, xc1, xc2, yc1, yc2,
                     omegar, omegal, targetroot, total_orbitals);
  // *************** END GRADIENTS CODE ************************* //
  free_tensors(d_f1, d_v2);
  free_tensors(d_t1,d_t2);
  free_vec_tensors(xc1,xc2,yc1,yc2);

  ec.flush_and_sync();
  MemoryManagerGA::destroy_coll(mgr);
//   delete ec;
}