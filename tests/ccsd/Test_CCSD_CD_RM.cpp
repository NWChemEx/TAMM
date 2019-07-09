// #define CATCH_CONFIG_RUNNER

#include "cd_ccsd_common.hpp"

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

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
    debug = ccsd_options.debug;
    if(rank == 0) ccsd_options.print();

    int maxiter    = ccsd_options.maxiter;
    double thresh  = ccsd_options.threshold;
    double zshiftl = 0.0;
    size_t ndiis   = 5;

    auto [MO,total_orbitals] = setupMOIS(ccsd_options.tilesize,
                    nao,ov_alpha,freeze_core,freeze_virtual);

    std::string out_fp = getfilename(filename)+"."+ccsd_options.basis;
    std::string files_dir = out_fp+"_files";
    std::string files_prefix = /*out_fp;*/ files_dir+"/"+out_fp;
    std::string f1file = files_prefix+".f1_mo";
    std::string t1file = files_prefix+".t1amp";
    std::string t2file = files_prefix+".t2amp";
    std::string v2file = files_prefix+".cholv2";
    std::string cholfile = files_prefix+".cholcount";
    std::string ccsdstatus = files_prefix+".ccsdstatus";
    
    bool ccsd_restart = ccsd_options.readt || 
        ( (fs::exists(t1file) && fs::exists(t2file)     
        && fs::exists(f1file) && fs::exists(v2file)) );

    //deallocates F_AO, C_AO
    auto [cholVpr,d_f1,chol_count, max_cvecs, CI] = cd_svd_ga_driver<T>
                        (options_map, ec, MO, AO_opt, ov_alpha, nao, freeze_core,
                                freeze_virtual, C_AO, F_AO, shells, shell_tile_map,
                                ccsd_restart, cholfile);

    TiledIndexSpace N = MO("all");

    auto [p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s] 
            = setupTensors(ec,MO,d_f1,ndiis);

    if(ccsd_restart) {
        read_from_disk(ec,d_f1,f1file);
        read_from_disk(ec,d_t1,t1file);
        read_from_disk(ec,d_t2,t2file);
        read_from_disk(ec,cholVpr,v2file);
        ec.pg().barrier();
        p_evl_sorted = tamm::diagonal(ec,d_f1());
    }
    
    else if(ccsd_options.writet) {
        // fs::remove_all(files_dir); 
        if(!fs::exists(files_dir)) fs::create_directories(files_dir);

        write_to_disk(ec,d_f1,f1file);
        write_to_disk(ec,cholVpr,v2file);

        if(rank==0){
          std::ofstream out(cholfile, std::ios::out);
          if(!out) cerr << "Error opening file " << cholfile << endl;
          out << chol_count << std::endl;
          out.close();
        }        
    }

    auto cc_t1 = std::chrono::high_resolution_clock::now();

    ccsd_restart = ccsd_restart && fs::exists(ccsdstatus);

    auto [residual, corr_energy] = cd_ccsd_driver<T>(
            ec, MO, CI, d_t1, d_t2, d_f1, 
            d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
            p_evl_sorted, 
            maxiter, thresh, zshiftl, ndiis, 
            2 * ov_alpha, cholVpr, ccsd_options.writet, ccsd_restart, files_prefix);

    ccsd_stats(ec, hf_energy,residual,corr_energy,thresh);

    if(ccsd_options.writet && !fs::exists(ccsdstatus)) {
        write_to_disk(ec,d_t1,t1file);
        write_to_disk(ec,d_t2,t2file);
        if(rank==0){
          std::ofstream out(ccsdstatus, std::ios::out);
          if(!out) cerr << "Error opening file " << ccsdstatus << endl;
          out << 1 << std::endl;
          out.close();
        }                
    }

    auto cc_t2 = std::chrono::high_resolution_clock::now();
    double ccsd_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(rank == 0) std::cout << std::endl << "Time taken for Cholesky CCSD: " << ccsd_time << " secs" << std::endl;

    free_tensors(d_r1, d_r2, d_t1, d_t2, d_f1, cholVpr);
    free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);

    ec.flush_and_sync();
    MemoryManagerGA::destroy_coll(mgr);
    // delete ec;

}