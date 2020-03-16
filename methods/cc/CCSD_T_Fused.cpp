// #define CATCH_CONFIG_RUNNER

#include "cd_ccsd_common.hpp"
#include "ccsd_t/ccsd_t_fused_driver.hpp"

void ccsd_driver();
std::string filename;
double ccsd_t_t2_GetTime = 0;
double ccsd_t_v2_GetTime = 0;
double genTime = 0;
double ccsd_t_data_per_rank = 0; //in GB

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

    //force writet on
    sys_data.options_map.ccsd_options.writet = true;

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

    TiledIndexSpace N = MO("all");

    #if 1
    //deallocates F_AO, C_AO
    auto [cholVpr,d_f1,chol_count, max_cvecs, CI] = cd_svd_ga_driver<T>
                        (sys_data, ec, MO, AO_opt, C_AO, F_AO, shells, shell_tile_map,
                                ccsd_restart, cholfile);

    auto [p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s] 
            = setupTensors(ec,MO,d_f1,ccsd_options.ndiis,ccsd_restart && fs::exists(ccsdstatus) && scf_conv);

    if(ccsd_restart) {
        read_from_disk(d_f1,f1file);
        if(fs::exists(t1file) && fs::exists(t2file)) {
            read_from_disk(d_t1,t1file);
            read_from_disk(d_t2,t2file);
        }
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

    ExecutionHW hw = ExecutionHW::CPU;

    #ifdef USE_TALSH_T
    hw = ExecutionHW::GPU;
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
        // write_to_disk(d_t1,t1file);
        // write_to_disk(d_t2,t2file);
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

    free_tensors(d_t1, d_t2, d_f1);
    ec.flush_and_sync();

    std::string fullV2file = files_prefix+".fullV2";
    bool  ccsd_t_restart =
        ( (fs::exists(t1file) && fs::exists(t2file)     
        && fs::exists(f1file) && fs::exists(fullV2file)) );

    Tensor<T> d_v2;
    if(!ccsd_t_restart) {
        d_v2 = setupV2<T>(ec,MO,CI,cholVpr,chol_count, hw);
        write_to_disk(d_v2,fullV2file);
        Tensor<T>::deallocate(d_v2);
    }

    Tensor<T>::deallocate(cholVpr);

    #ifdef USE_TALSH_T
    //talshStats();
    if(has_gpu) talsh_instance.shutdown();
    #endif  

    #endif 

    #if 0
    TiledIndexSpace O = MO("occ");
    TiledIndexSpace V = MO("virt");
    double residual=0, corr_energy=0;

    Tensor<T> d_f1{{N,N},{1,1}};
    Tensor<T> d_t1{{V,O},{1,1}};
    Tensor<T> d_t2{{V,V,O,O},{2,2}};

    {
        genTime = 0;
        TimerGuard tg_total{&genTime};
        Tensor<T>::allocate(&ec,d_f1);
        Scheduler{ec}(d_f1()=4.20).execute();
    }
    if(rank==0) std::cout << "done alloc/init F1: " << genTime << "secs" << std::endl; 

    std::vector<T> p_evl_sorted;
    {
        genTime = 0;
        TimerGuard tg_total{&genTime};
        p_evl_sorted = tamm::diagonal(d_f1);
    }
    if(rank==0) std::cout << "diagonal(f1): " << genTime << "secs" << std::endl; 

    {
        genTime = 0;
        TimerGuard tg_total{&genTime};
        Tensor<T>::deallocate(C_AO,F_AO);
    }
    if(rank==0) std::cout << "deallocate(C_AO,F_AO): " << genTime << "secs" << std::endl; 

    Tensor<T> d_v2{{N,N,N,N},{2,2}};
    {
      genTime = 0; memTime1 =0; memTime2 = 0; memTime3=0; memTime4=0;
      memTime5=0; memTime6=0; memTime7=0; memTime8=0;
        TimerGuard tg_total{&genTime};    
        Tensor<T>::allocate(&ec,d_v2);
    }
    if(rank==0) {
        std::cout << "Time to allocate V2: " << genTime << "secs" << std::endl; 
        std::cout << "   -> timpl: allocate: total: " << memTime1 << "secs" << std::endl; 
        std::cout << "   -> timpl: allocate: dist clone: " << memTime2 << "secs" << std::endl; 
        std::cout << "   -> mmga: alloc_coll: total: " << memTime3 << "secs" << std::endl; 
        std::cout << "   -> mmga: alloc_coll: ga-create-irreg: " << memTime4 << "secs" << std::endl; 
        std::cout << "   -> mmga: alloc_coll: all_reduce: " << memTime5 << "secs" << std::endl; 
        std::cout << "   -> mmga: alloc_coll: all_gather: " << memTime6 << "secs" << std::endl; 
        std::cout << "   -> mmga: alloc_coll: partial_sum: " << memTime7 << "secs" << std::endl; 
        std::cout << "   -> mmga: alloc_coll: ga-dist: " << memTime8 << "secs" << std::endl; 
    }

    {
       genTime = 0;
       TimerGuard tg_total{&genTime};    
       Tensor<T>::allocate(&ec,d_t1,d_t2);
    }
    if(rank==0) 
        std::cout << "allocate T1,T2: " << genTime << "secs" << std::endl; 

    auto cc_t1 = std::chrono::high_resolution_clock::now();

    // Scheduler{ec}
    // (d_t1() = 5.21)
    // (d_v2() = 2.3)
    // (d_t2() = 8.234)
    // .execute();
    auto cc_t2= std::chrono::high_resolution_clock::now();
    double alloc_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();    
    if(rank==0) std::cout << "initialize T1,T2,V2: " << alloc_time << "secs" << std::endl; 

    #endif

    cc_t1 = std::chrono::high_resolution_clock::now();
    double energy1=0, energy2=0;

    if(rank==0) {
        auto mo_tiles = MO.input_tile_sizes();
        cout << endl << "CCSD MO Tiles = " << mo_tiles << endl;   
    }

    auto [MO1,total_orbitals1] = setupMOIS(sys_data,true);
    TiledIndexSpace N1 = MO1("all");
    TiledIndexSpace O1 = MO1("occ");
    TiledIndexSpace V1 = MO1("virt");     

    Tensor<T> t_d_f1{{N1,N1},{1,1}};
    Tensor<T> t_d_t1{{V1,O1},{1,1}};
    Tensor<T> t_d_t2{{V1,V1,O1,O1},{2,2}};
    Tensor<T> t_d_v2{{N1,N1,N1,N1},{2,2}};
    Tensor<T>::allocate(&ec,t_d_f1,t_d_t1,t_d_t2,t_d_v2);

    Scheduler{ec}   
    (t_d_f1() = 0)
    (t_d_t1() = 0)
    (t_d_t2() = 0)
    (t_d_v2() = 0)
    .execute();

    TiledIndexSpace O = MO("occ");
    TiledIndexSpace V = MO("virt");
    Tensor<T> wd_f1{{N,N},{1,1}};
    Tensor<T> wd_t1{{V,O},{1,1}};
    Tensor<T> wd_t2{{V,V,O,O},{2,2}};
    Tensor<T> wd_v2{{N,N,N,N},{2,2}};

    read_from_disk(t_d_f1,f1file,false,wd_f1);
    read_from_disk(t_d_t1,t1file,false,wd_t1);
    read_from_disk(t_d_t2,t2file,false,wd_t2);
    read_from_disk(t_d_v2,fullV2file,false,wd_v2); 

    ec.pg().barrier();
    p_evl_sorted = tamm::diagonal(t_d_f1);

    cc_t1 = std::chrono::high_resolution_clock::now();

    Index noab=MO1("occ").num_tiles();
    Index nvab=MO1("virt").num_tiles();
    std::vector<int> k_spin;
    for(tamm::Index x=0;x<noab/2;x++) k_spin.push_back(1);
    for(tamm::Index x=noab/2;x<noab;x++) k_spin.push_back(2);
    for(tamm::Index x=0;x<nvab/2;x++) k_spin.push_back(1);
    for(tamm::Index x=nvab/2;x<nvab;x++) k_spin.push_back(2);

    if(rank==0) cout << endl << "Running CCSD(T) calculation" << endl;

    bool is_restricted = true;
    if(sys_data.options_map.scf_options.scf_type == "uhf") is_restricted = false;

    std::tie(energy1,energy2) = ccsd_t_fused_driver<T>(sys_data,ec,k_spin,MO1,t_d_t1,t_d_t2,t_d_v2,
                                    p_evl_sorted,hf_energy+corr_energy,ccsd_options.icuda,is_restricted);

    double g_energy1,g_energy2;
    MPI_Reduce(&energy1, &g_energy1, 1, MPI_DOUBLE, MPI_SUM, 0, ec.pg().comm());
    MPI_Reduce(&energy2, &g_energy2, 1, MPI_DOUBLE, MPI_SUM, 0, ec.pg().comm());
    energy1 = g_energy1;
    energy2 = g_energy2;

    if (rank==0 && energy1!=-999){

        std::cout.precision(15);
        cout << "CCSD[T] correction energy / hartree  = " << energy1 << endl;
        cout << "CCSD[T] correlation energy / hartree = " << corr_energy + energy1 << endl;
        cout << "CCSD[T] total energy / hartree       = " << hf_energy + corr_energy + energy1 << endl;

        cout << "CCSD(T) correction energy / hartree  = " << energy2 << endl;
        cout << "CCSD(T) correlation energy / hartree = " << corr_energy + energy2 << endl;
        cout << "CCSD(T) total energy / hartree       = " << hf_energy + corr_energy + energy2 << endl;
    
    }

    free_tensors(t_d_t1, t_d_t2, t_d_f1, t_d_v2);

    ec.flush_and_sync();
    MemoryManagerGA::destroy_coll(mgr);
    // delete ec;

    cc_t2 = std::chrono::high_resolution_clock::now();
    auto ccsd_t_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();


    double g_ccsd_t_time,g_ccsd_t_data_per_rank;
    double g_t2_getTime,min_t2_getTime,max_t2_getTime, g_v2_getTime,min_v2_getTime,max_v2_getTime;

    MPI_Reduce(&ccsd_t_time, &g_ccsd_t_time, 1, MPI_DOUBLE, MPI_SUM, 0, ec.pg().comm());

    MPI_Reduce(&ccsd_t_t2_GetTime, &g_t2_getTime,   1, MPI_DOUBLE, MPI_SUM, 0, ec.pg().comm());
    MPI_Reduce(&ccsd_t_t2_GetTime, &min_t2_getTime, 1, MPI_DOUBLE, MPI_MIN, 0, ec.pg().comm());
    MPI_Reduce(&ccsd_t_t2_GetTime, &max_t2_getTime, 1, MPI_DOUBLE, MPI_MAX, 0, ec.pg().comm());

    MPI_Reduce(&ccsd_t_v2_GetTime, &g_v2_getTime,   1, MPI_DOUBLE, MPI_SUM, 0, ec.pg().comm());
    MPI_Reduce(&ccsd_t_v2_GetTime, &min_v2_getTime, 1, MPI_DOUBLE, MPI_MIN, 0, ec.pg().comm());
    MPI_Reduce(&ccsd_t_v2_GetTime, &max_v2_getTime, 1, MPI_DOUBLE, MPI_MAX, 0, ec.pg().comm());    

    ccsd_t_data_per_rank = (ccsd_t_data_per_rank * 8.0) / (1024*1024.0*1024); //GB
    MPI_Reduce(&ccsd_t_data_per_rank, &g_ccsd_t_data_per_rank, 1, MPI_DOUBLE, MPI_SUM, 0, ec.pg().comm());

    if(rank == 0 && energy1!=-999) {
        auto nranks = ec.pg().size().value();
        // cout << "nranks,ngpus = " << nranks << endl;
        ccsd_t_time = g_ccsd_t_time/nranks;

        auto print_profile_stats = [&](const std::string& timer_type, const double g_tval, const double tval_min, const double tval_max){
            const double tval = g_tval/nranks;
            std::cout.precision(3);
            std::cout << "   -> " << timer_type << ": " << tval << "s (" << tval*100.0/ccsd_t_time << "%), (min,max) = (" << tval_min << "," << tval_max << ")" << std::endl;
        };

        std::cout << "\nTotal CCSD(T) Time: " << ccsd_t_time << " secs\n";
        print_profile_stats("T2 GetTime", g_t2_getTime, min_t2_getTime, max_t2_getTime);
        print_profile_stats("V2 GetTime", g_v2_getTime, min_v2_getTime, max_v2_getTime);
        std::cout << "   -> Data Transfer (GB): " << g_ccsd_t_data_per_rank/nranks << std::endl;
    }


}
