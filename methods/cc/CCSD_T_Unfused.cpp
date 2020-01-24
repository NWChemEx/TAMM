// #define CATCH_CONFIG_RUNNER

#include "cd_ccsd_common.hpp"
#include "ccsd_t/ccsd_t_unfused_driver.hpp"

void ccsd_driver();
std::string filename; //bad, but no choice
bool use_nwc_gpu_kernels = false;

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

    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    RuntimeEngine re;
    ExecutionContext ec{pg, &distribution, mgr, &re};
    auto rank = ec.pg().rank();

    auto [sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, AO_opt, AO_tis,scf_conv]  
                    = hartree_fock_driver<T>(ec,filename);

    CCSDOptions ccsd_options = sys_data.options_map.ccsd_options;
    debug = ccsd_options.debug;
    if(rank == 0) ccsd_options.print();

    if(rank==0) cout << endl << "#occupied, #virtual = " << sys_data.nocc << ", " << sys_data.nvir << endl;
    
    auto [MO,total_orbitals] = setupMOIS(sys_data);

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
                        (sys_data, ec, MO, AO_opt, C_AO, F_AO, shells, shell_tile_map,
                                ccsd_restart, cholfile);

    TiledIndexSpace N = MO("all");

    auto [p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s] 
            = setupTensors(ec,MO,d_f1,ccsd_options.ndiis,ccsd_restart && fs::exists(ccsdstatus) && scf_conv);

    if(ccsd_restart) {
        read_from_disk(d_f1,f1file);
        read_from_disk(d_t1,t1file);
        read_from_disk(d_t2,t2file);
        read_from_disk(cholVpr,v2file);
        ec.pg().barrier();
        p_evl_sorted = tamm::diagonal(d_f1);
    }
    
    else if(ccsd_options.writet) {
        // fs::remove_all(files_dir); 
        if(!fs::exists(files_dir)) fs::create_directories(files_dir);

        write_to_disk(d_f1,f1file);
        write_to_disk(cholVpr,v2file);

        if(rank==0){
          std::ofstream out(cholfile, std::ios::out);
          if(!out) cerr << "Error opening file " << cholfile << endl;
          out << chol_count << std::endl;
          out.close();
        }        
    }

    if(rank==0 && debug){
      cout << "eigen values:" << endl << std::string(50,'-') << endl;
      for (size_t i=0;i<p_evl_sorted.size();i++) cout << i+1 << "   " << p_evl_sorted[i] << endl;
      cout << std::string(50,'-') << endl;
    }
    
    ec.pg().barrier();

    #ifdef USE_TALSH_T
    const bool has_gpu = ec.has_gpu();
    TALSH talsh_instance;
    if(has_gpu) talsh_instance.initialize(ec.gpu_devid(),rank.value());
    #endif

    auto cc_t1 = std::chrono::high_resolution_clock::now();

    ccsd_restart = ccsd_restart && fs::exists(ccsdstatus) && scf_conv;

    auto [residual, corr_energy] = cd_ccsd_driver<T>(
            sys_data, ec, MO, CI, d_t1, d_t2, d_f1, 
            d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
            p_evl_sorted, 
            cholVpr, ccsd_restart, files_prefix);

    ccsd_stats(ec, hf_energy,residual,corr_energy,ccsd_options.threshold);

    auto cc_t2 = std::chrono::high_resolution_clock::now();

    if(ccsd_options.writet && !fs::exists(ccsdstatus)) {
        write_to_disk(d_t1,t1file);
        write_to_disk(d_t2,t2file);
        if(rank==0){
          std::ofstream out(ccsdstatus, std::ios::out);
          if(!out) cerr << "Error opening file " << ccsdstatus << endl;
          out << 1 << std::endl;
          out.close();
        }                
    }

    double ccsd_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(rank == 0) std::cout << "\nTime taken for Cholesky CCSD: " << ccsd_time << " secs\n";

    if(!ccsd_restart) {
        free_tensors(d_r1,d_r2);
        free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);
    }

    ec.flush_and_sync();

    Tensor<T> d_v2 = setupV2<T>(ec,MO,CI,cholVpr,chol_count, ExecutionHW::CPU);
    Tensor<T>::deallocate(cholVpr);

    #ifdef USE_TALSH_T
    //talshStats();
    if(has_gpu) talsh_instance.shutdown();
    #endif  

    cc_t1 = std::chrono::high_resolution_clock::now();

    Index noab=MO("occ").num_tiles();
    Index nvab=MO("virt").num_tiles();
    std::vector<int> k_spin;
    for(auto x=0;x<noab/2;x++) k_spin.push_back(1);
    for(auto x=noab/2;x<noab;x++) k_spin.push_back(2);
    for(auto x=0;x<nvab/2;x++) k_spin.push_back(1);
    for(auto x=nvab/2;x<nvab;x++) k_spin.push_back(2);

    if(rank==0) cout << "\nCCSD(T)\n";

    bool is_restricted = true;
    if(sys_data.options_map.scf_options.scf_type == "uhf") is_restricted = false;

    auto [energy1,energy2] = ccsd_t_unfused_driver(ec,k_spin,MO,d_t1,d_t2,d_v2,
                p_evl_sorted,hf_energy+corr_energy,ccsd_options.icuda,is_restricted,use_nwc_gpu_kernels);

    double g_energy1,g_energy2;
    MPI_Reduce(&energy1, &g_energy1, 1, MPI_DOUBLE, MPI_SUM, 0, ec.pg().comm());
    MPI_Reduce(&energy2, &g_energy2, 1, MPI_DOUBLE, MPI_SUM, 0, ec.pg().comm());
    energy1 = g_energy1;
    energy2 = g_energy2;

    if (rank==0 && energy1!=-999){

        std::cout.precision(15);
        cout << "CCSD[T] correction energy / hartree  = " << energy1 << endl;
        cout << "CCSD[T] correlation energy / hartree = " << corr_energy+energy1 << endl;
        cout << "CCSD[T] total energy / hartree       = " << hf_energy+corr_energy+energy1 << endl;

        cout << "CCSD(T) correction energy / hartree  = " << energy2 << endl;
        cout << "CCSD(T) correlation energy / hartree = " << corr_energy+energy2 << endl;
        cout << "CCSD(T) total energy / hartree       = " << hf_energy+corr_energy+energy2 << endl;
    
    }

    free_tensors(d_t1, d_t2, d_f1, d_v2);

    ec.flush_and_sync();
    MemoryManagerGA::destroy_coll(mgr);
    // delete ec;

    cc_t2 = std::chrono::high_resolution_clock::now();
    auto ccsd_t_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(rank == 0 && energy1!=-999) std::cout << "\nTime taken for CCSD(T): " << ccsd_t_time << " secs\n";


}
