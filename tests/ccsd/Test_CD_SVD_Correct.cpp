// #define CATCH_CONFIG_RUNNER

#include "ccsd_util.hpp"
#include "cd_svd_correct.hpp"

using namespace tamm;

void cd_svd_driver();

template<typename T> 
std::tuple<Tensor<T>,Tensor<T>,TAMM_SIZE, tamm::Tile>  cd_svd_driver_correct(OptionsMap options_map,
 ExecutionContext& ec, TiledIndexSpace& MO, TiledIndexSpace& AO_tis,
  const TAMM_SIZE ov_alpha, const TAMM_SIZE nao, const TAMM_SIZE freeze_core,
  const TAMM_SIZE freeze_virtual, Tensor<TensorType> C_AO, Tensor<TensorType> F_AO,
  libint2::BasisSet& shells, std::vector<size_t>& shell_tile_map){

    CDOptions cd_options = options_map.cd_options;
    tamm::Tile max_cvecs = cd_options.max_cvecs_factor * nao;
    auto diagtol = cd_options.diagtol; // tolerance for the max. diagonal

    std::cout << std::defaultfloat;
    auto rank = ec.pg().rank();
    if(rank==0) cd_options.print();

    TiledIndexSpace N = MO("all");

    Tensor<T> d_f1{{N,N},{1,1}};
    Tensor<T>::allocate(&ec,d_f1);

    auto hf_t1        = std::chrono::high_resolution_clock::now();
    TAMM_SIZE chol_count = 0;

    //std::tie(V2) = 
    Tensor<T> cholVpr = cd_svd_correct(ec, MO, AO_tis, ov_alpha, nao, freeze_core, freeze_virtual,
                                C_AO, F_AO, d_f1, chol_count, max_cvecs, diagtol, shells, shell_tile_map);
    auto hf_t2        = std::chrono::high_resolution_clock::now();
    double cd_svd_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    if(rank == 0) std::cout << "\nTotal Time taken for CD (+SVD): " << cd_svd_time
              << " secs\n";

    Tensor<T>::deallocate(C_AO,F_AO);

    return std::make_tuple(cholVpr, d_f1, chol_count, max_cvecs);

}

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

    cd_svd_driver();
    
    GA_Terminate();
    MPI_Finalize();

    return 0;
}


void cd_svd_driver() {

    using T = double;

    ProcGroup pg{GA_MPI_Comm()};
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    ExecutionContext ec{pg, &distribution, mgr};

    //TODO: read from input file, assume no freezing for now
    TAMM_SIZE freeze_core    = 0;
    TAMM_SIZE freeze_virtual = 0;

    auto [options_map, ov_alpha, nao, hf_energy, shells, shell_tile_map, C_AO, F_AO, AO_opt, AO_tis] 
                    = hartree_fock_driver<T>(ec,filename);

    auto [MO,total_orbitals] = setupMOIS(options_map.ccsd_options.tilesize,
                    nao,ov_alpha,freeze_core,freeze_virtual);

    //deallocates F_AO, C_AO
    auto [cholVpr,d_f1,chol_count, max_cvecs] = cd_svd_driver_correct<T>
                        (options_map, ec, MO, AO_opt, ov_alpha, nao, freeze_core,
                                freeze_virtual, C_AO, F_AO, shells, shell_tile_map);

    Tensor<T>::deallocate(d_f1,cholVpr);

    ec.flush_and_sync();
    MemoryManagerGA::destroy_coll(mgr);
    // delete ec;

}
