// #define CATCH_CONFIG_RUNNER

#include "cd_ccsd_common.hpp"
#include "ccsd_t/ccsd_t_gpu_tgen.hpp"

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

    int maxiter    = ccsd_options.ccsd_maxiter;
    double thresh  = ccsd_options.threshold;
    double zshiftl = 0.0;
    size_t ndiis   = 5;

    const TAMM_SIZE nocc = 2 * ov_alpha;
    const TAMM_SIZE nvir = 2*nao - 2*ov_alpha;
    if(rank==0) cout << endl << "#occupied, #virtual = " << nocc << ", " << nvir << endl;
    
    auto [MO,total_orbitals] = setupMOIS(ccsd_options.tilesize,
                    nao,ov_alpha,freeze_core,freeze_virtual);

    //deallocates F_AO, C_AO
    auto [cholVpr,d_f1,chol_count, max_cvecs, CI] = cd_svd_ga_driver<T>
                        (options_map, ec, MO, AO_opt, ov_alpha, nao, freeze_core,
                                freeze_virtual, C_AO, F_AO, shells, shell_tile_map);

    TiledIndexSpace N = MO("all");

    auto [p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s] 
            = setupTensors(ec,MO,d_f1,ndiis);

    #ifdef USE_TALSH_T
    TALSH talsh_instance;
    talsh_instance.initialize(rank.value());
    #endif

    auto cc_t1 = std::chrono::high_resolution_clock::now();

    auto [residual, corr_energy] = cd_ccsd_driver<T>(
            ec, MO, CI, d_t1, d_t2, d_f1, 
            d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
            p_evl_sorted, 
            maxiter, thresh, zshiftl, ndiis, 
            2 * ov_alpha, cholVpr);

    ccsd_stats(ec, hf_energy,residual,corr_energy,thresh);

    auto cc_t2 = std::chrono::high_resolution_clock::now();
    double ccsd_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(rank == 0) std::cout << "\nTime taken for Cholesky CCSD: " << ccsd_time << " secs\n";

    #ifdef USE_TALSH_T
    //talshStats();
    talsh_instance.shutdown();
    #endif  

    free_tensors(d_r1, d_r2, d_f1);
    free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);

    ec.flush_and_sync();

    Tensor<T> d_v2 = setupV2<T>(ec,MO,CI,cholVpr,chol_count, total_orbitals, ov_alpha, nao - ov_alpha);
    Tensor<T>::deallocate(cholVpr);

    cc_t1 = std::chrono::high_resolution_clock::now();

    auto n_alpha = ov_alpha;
    TAMM_SIZE n_beta{nao - ov_alpha};

    Index noab=MO("occ").num_tiles();
    Index nvab=MO("virt").num_tiles();
    std::vector<int> k_spin;
    for(auto x=0;x<noab/2;x++) k_spin.push_back(1);
    for(auto x=noab/2;x<noab;x++) k_spin.push_back(2);
    for(auto x=0;x<nvab/2;x++) k_spin.push_back(1);
    for(auto x=nvab/2;x<nvab;x++) k_spin.push_back(2);

    if(rank==0) cout << "\nCCSD(T)\n";

    auto [energy1,energy2] = ccsd_t_tgen_driver<T>(ec,k_spin,n_alpha,n_beta,MO,d_t1,d_t2,d_v2,
                p_evl_sorted,hf_energy+corr_energy,ccsd_options.icuda);

    if (rank==0 && energy1!=-999){

        cout << "CCSD[T] correction energy / hartree  = " << energy1 << endl;
        cout << "CCSD[T] correlation energy / hartree = " << corr_energy+energy1 << endl;
        cout << "CCSD[T] total energy / hartree       = " << hf_energy+corr_energy+energy1 << endl;

        cout << "CCSD(T) correction energy / hartree  = " << energy2 << endl;
        cout << "CCSD(T) correlation energy / hartree = " << corr_energy+energy2 << endl;
        cout << "CCSD(T) total energy / hartree       = " << hf_energy+corr_energy+energy2 << endl;
    
    }

    free_tensors(d_t1, d_t2, d_v2);

    ec.flush_and_sync();
    MemoryManagerGA::destroy_coll(mgr);
    // delete ec;

    cc_t2 = std::chrono::high_resolution_clock::now();
    ccsd_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(rank == 0 && energy1!=-999) std::cout << "\nTime taken for CCSD(T): " << ccsd_time << " secs\n";


}
