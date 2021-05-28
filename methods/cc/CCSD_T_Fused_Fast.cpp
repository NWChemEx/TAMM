#include "cd_ccsd_os_ann.hpp"
#include "ccsd_t/ccsd_t_fused_driver.hpp"

void ccsd_t_driver();
std::string filename;
double ccsdt_s1_t1_GetTime = 0;
double ccsdt_s1_v2_GetTime = 0;
double ccsdt_d1_t2_GetTime = 0;
double ccsdt_d1_v2_GetTime = 0;
double ccsdt_d2_t2_GetTime = 0;
double ccsdt_d2_v2_GetTime = 0;
double genTime = 0;
double ccsd_t_data_per_rank = 0; //in GB

int main( int argc, char* argv[] )
{
    if(argc<2){
        std::cout << "Please provide an input file!" << std::endl;
        return 1;
    }

    filename = std::string(argv[1]);
    std::ifstream testinput(filename); 
    if(!testinput){
        std::cout << "Input file provided [" << filename << "] does not exist!" << std::endl;
        return 1;
    }

    tamm::initialize(argc, argv);

    ccsd_t_driver();

    tamm::finalize();

    return 0;
}

void ccsd_t_driver() {

    // std::cout << "Input file provided = " << filename << std::endl;

    using T = double;

    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    auto rank = ec.pg().rank();

    auto [sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, C_beta_AO, F_beta_AO, AO_opt, AO_tis,scf_conv]  
                    = hartree_fock_driver<T>(ec,filename);

    int nsranks = sys_data.nbf/15;
    if(nsranks < 1) nsranks=1;
    int ga_cnn = GA_Cluster_nnodes();
    if(nsranks>ga_cnn) nsranks=ga_cnn;
    nsranks = nsranks * GA_Cluster_nprocs(0);
    int subranks[nsranks];
    for (int i = 0; i < nsranks; i++) subranks[i] = i;
    auto world_comm = ec.pg().comm();
    MPI_Group world_group;
    MPI_Comm_group(world_comm,&world_group);
    MPI_Group subgroup;
    MPI_Group_incl(world_group,nsranks,subranks,&subgroup);
    MPI_Comm subcomm;
    MPI_Comm_create(world_comm,subgroup,&subcomm);
    
    ProcGroup sub_pg;
    ExecutionContext *sub_ec=nullptr;

    if(subcomm != MPI_COMM_NULL){
        sub_pg = ProcGroup::create_coll(subcomm);
        sub_ec = new ExecutionContext(sub_pg, DistributionKind::nw, MemoryManagerKind::ga);
    }

    Scheduler sub_sch{*sub_ec};

    //force writet on
    sys_data.options_map.ccsd_options.writet = true;
    sys_data.options_map.ccsd_options.computeTData = true;

    CCSDOptions& ccsd_options = sys_data.options_map.ccsd_options;
    debug = ccsd_options.debug;
    if(rank == 0) ccsd_options.print();
    
    if(rank==0) cout << endl << "#occupied, #virtual = " << sys_data.nocc << ", " << sys_data.nvir << endl;
    
    auto [MO,total_orbitals] = setupMOIS(sys_data,true);

    std::string out_fp = sys_data.output_file_prefix+"."+ccsd_options.basis;
    std::string files_dir = out_fp+"_files";
    std::string files_prefix = /*out_fp;*/ files_dir+"/"+out_fp;
    std::string f1file = files_prefix+".f1_mo";
    std::string t1file = files_prefix+".t1amp";
    std::string t2file = files_prefix+".t2amp";
    std::string v2file = files_prefix+".cholv2";
    std::string cholfile = files_prefix+".cholcount";
    std::string ccsdstatus = files_prefix+".ccsdstatus";

    const bool is_rhf = (sys_data.scf_type == sys_data.SCFType::rhf);

    bool ccsd_restart = ccsd_options.readt || 
        ( (fs::exists(t1file) && fs::exists(t2file)     
        && fs::exists(f1file) && fs::exists(v2file)) );

    TiledIndexSpace N = MO("all");

    #if 0
    //deallocates F_AO, C_AO
    auto [cholVpr,d_f1,lcao,chol_count, max_cvecs, CI] = cd_svd_ga_driver<T>
                        (sys_data, ec, MO, AO_opt, C_AO, F_AO, C_beta_AO, F_beta_AO, shells, shell_tile_map,
                                ccsd_restart, cholfile);
    free_tensors(lcao);

    if(ccsd_options.writev) ccsd_options.writet = true;

    TiledIndexSpace N = MO("all");

    std::vector<T> p_evl_sorted;
    Tensor<T> d_r1, d_r2, d_t1, d_t2;
    std::vector<Tensor<T>> d_r1s, d_r2s, d_t1s, d_t2s;

    if(is_rhf) 
        std::tie(p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s)
                = setupTensors_cs(ec,MO,d_f1,ccsd_options.ndiis,ccsd_restart && fs::exists(ccsdstatus) && scf_conv);
    else
        std::tie(p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s)
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

    auto cc_t1 = std::chrono::high_resolution_clock::now();

    ExecutionHW ex_hw = ExecutionHW::CPU;
    #ifdef USE_TALSH_T
    ex_hw = ExecutionHW::GPU;
    const bool has_gpu = ec.has_gpu();
    TALSH talsh_instance;
    if(has_gpu) talsh_instance.initialize(ec.gpu_devid(),rank.value());
    #endif

    ccsd_restart = ccsd_restart && fs::exists(ccsdstatus) && scf_conv;

    std::string fullV2file = files_prefix+".fullV2";
    t1file = files_prefix+".fullT1amp";
    t2file = files_prefix+".fullT2amp";

    bool  computeTData = ccsd_options.computeTData; 
    if(ccsd_options.writev) 
        computeTData = computeTData && !fs::exists(fullV2file) 
                && !fs::exists(t1file) && !fs::exists(t2file);

    if(computeTData && is_rhf) {
        TiledIndexSpace O = MO("occ");
        TiledIndexSpace V = MO("virt");

        const int otiles = O.num_tiles();
        const int vtiles = V.num_tiles();
        const int obtiles = MO("occ_beta").num_tiles();
        const int vbtiles = MO("virt_beta").num_tiles();

        o_beta = {MO("occ"), range(obtiles,otiles)};
        v_beta = {MO("virt"), range(vbtiles,vtiles)};

        dt1_full = {{V,O},{1,1}};
        dt2_full = {{V,V,O,O},{2,2}};
        t1_bb    = {{v_beta ,o_beta}                 ,{1,1}};
        t2_bbbb  = {{v_beta ,v_beta ,o_beta ,o_beta} ,{2,2}};

        Tensor<T>::allocate(&ec,t1_bb,t2_bbbb,dt1_full,dt2_full);
        // (dt1_full() = 0)
        // (dt1_full() = 0)
    }

    double residual=0, corr_energy=0;

    if(is_rhf) {
      if(ccsd_restart) {
          if(subcomm != MPI_COMM_NULL) {
              const int ppn = GA_Cluster_nprocs(0);
              if(rank==0) std::cout << "Executing with " << nsranks << " ranks (" << nsranks/ppn << " nodes)" << std::endl; 
              std::tie(residual, corr_energy) = cd_ccsd_cs_driver<T>(
                      sys_data, *sub_ec, MO, CI, d_t1, d_t2, d_f1, 
                      d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
                      p_evl_sorted, 
                      cholVpr, ccsd_restart, files_prefix,
                      computeTData);
          }
          ec.pg().barrier();
      }
      else {
          std::tie(residual, corr_energy) = cd_ccsd_cs_driver<T>(
                  sys_data, ec, MO, CI, d_t1, d_t2, d_f1, 
                  d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
                  p_evl_sorted, 
                  cholVpr, ccsd_restart, files_prefix,
                  computeTData);
          }      
    }
    else
        std::tie(residual, corr_energy) = cd_ccsd_os_driver<T>(
            sys_data, ec, MO, CI, d_t1, d_t2, d_f1, 
            d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
            p_evl_sorted, 
            cholVpr, ccsd_restart, files_prefix,
            computeTData);

    if(computeTData && is_rhf) {
        free_tensors(t1_bb,t2_bbbb);
        if(ccsd_options.writev) {
            write_to_disk(dt1_full,t1file);
            write_to_disk(dt2_full,t2file);
            free_tensors(dt1_full, dt2_full);
        }
    }

    ccsd_stats(ec, hf_energy,residual,corr_energy,ccsd_options.threshold);

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

    if(subcomm != MPI_COMM_NULL){
      (*sub_ec).flush_and_sync();
      MPI_Comm_free(&subcomm);
    }

    auto cc_t2 = std::chrono::high_resolution_clock::now();
    double ccsd_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(rank == 0) { 
      if(is_rhf)
        std::cout << std::endl << "Time taken for Closed Shell Cholesky CCSD: " << ccsd_time << " secs" << std::endl;
      else
        std::cout << std::endl << "Time taken for Open Shell Cholesky CCSD: " << ccsd_time << " secs" << std::endl;
    }

    double printtol=ccsd_options.printtol;
    if (rank == 0 && debug) {
        std::cout << std::endl << "Threshold for printing amplitudes set to: " << printtol << std::endl;
        std::cout << "T1 amplitudes" << std::endl;
        print_max_above_threshold(d_t1,printtol);
        std::cout << "T2 amplitudes" << std::endl;
        print_max_above_threshold(d_t2,printtol);
    }

    if(!ccsd_restart) {
        free_tensors(d_r1,d_r2);
        free_vec_tensors(d_r1s, d_r2s, d_t1s, d_t2s);
    }

    if(is_rhf) free_tensors(d_t1, d_t2);
    ec.flush_and_sync();

    bool  ccsd_t_restart = fs::exists(t1file) && fs::exists(t2file) &&
                           fs::exists(f1file) && fs::exists(fullV2file);

    Tensor<T> d_v2;
    if(computeTData) {
        d_v2 = setupV2<T>(ec,MO,CI,cholVpr,chol_count, ex_hw);
        if(ccsd_options.writev) {
          write_to_disk(d_v2,fullV2file,true);
          Tensor<T>::deallocate(d_v2);
        }
    }

    free_tensors(cholVpr);

    #ifdef USE_TALSH_T
    //talshStats();
    if(has_gpu) talsh_instance.shutdown();
    #endif  

    #endif 

    #if 1
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

    double energy1=0, energy2=0;

    if(rank==0) {
        auto mo_tiles = MO.input_tile_sizes();
        cout << endl << "CCSD MO Tiles = " << mo_tiles << endl;   
    }

    auto [MO1,total_orbitals1] = setupMOIS(sys_data,true);
    TiledIndexSpace N1 = MO1("all");
    TiledIndexSpace O1 = MO1("occ");
    TiledIndexSpace V1 = MO1("virt");     

    #if 0

    // Tensor<T> t_d_f1{{N1,N1},{1,1}};
    Tensor<T> t_d_t1{{V1,O1},{1,1}};
    Tensor<T> t_d_t2{{V1,V1,O1,O1},{2,2}};
    Tensor<T> t_d_v2{{N1,N1,N1,N1},{2,2}};
    Tensor<T>::allocate(&ec,t_d_t1,t_d_t2,t_d_v2);

    if(!ccsd_t_restart) {
        if(!is_rhf) {
          dt1_full = d_t1;
          dt2_full = d_t2;
        }        
        if(rank==0) {
            cout << endl << "Retile T1,T2,V2 ... " << endl;   
        }

        Scheduler{ec}   
        // (t_d_f1() = 0)
    (t_d_t1() = 0)
    (t_d_t2() = 0)
    (t_d_v2() = 0)
    .execute();

    TiledIndexSpace O = MO("occ");
    TiledIndexSpace V = MO("virt");

        if(ccsd_options.writev) {
          // Tensor<T> wd_f1{{N,N},{1,1}};
          Tensor<T> wd_t1{{V,O},{1,1}};
          Tensor<T> wd_t2{{V,V,O,O},{2,2}};
          Tensor<T> wd_v2{{N,N,N,N},{2,2}};
                        
          // read_from_disk(t_d_f1,f1file,false,wd_f1);
          read_from_disk(t_d_t1,t1file,false,wd_t1);
          read_from_disk(t_d_t2,t2file,false,wd_t2);
          read_from_disk(t_d_v2,fullV2file,false,wd_v2); 
          
          ec.pg().barrier();
          // write_to_disk(t_d_f1,f1file);
          write_to_disk(t_d_t1,t1file);
          write_to_disk(t_d_t2,t2file);
          write_to_disk(t_d_v2,fullV2file);
        }
        
        else {
          retile_tamm_tensor(dt1_full,t_d_t1);
          retile_tamm_tensor(dt2_full,t_d_t2);
          if(is_rhf) free_tensors(dt1_full, dt2_full);
          retile_tamm_tensor(d_v2,t_d_v2,"V2");
          free_tensors(d_v2);
        }        
    }
    else if(ccsd_options.writev) {
        // read_from_disk(t_d_f1,f1file);
        read_from_disk(t_d_t1,t1file);
        read_from_disk(t_d_t2,t2file);
        read_from_disk(t_d_v2,fullV2file);
    }

    if(!is_rhf) free_tensors(d_t1, d_t2);

    p_evl_sorted = tamm::diagonal(d_f1);
    #endif

    cc_t1 = std::chrono::high_resolution_clock::now();

    Index noab=MO("occ").num_tiles();
    Index nvab=MO("virt").num_tiles();
    std::vector<int> k_spin;
    for(tamm::Index x=0;x<noab/2;x++) k_spin.push_back(1);
    for(tamm::Index x=noab/2;x<noab;x++) k_spin.push_back(2);
    for(tamm::Index x=0;x<nvab/2;x++) k_spin.push_back(1);
    for(tamm::Index x=nvab/2;x<nvab;x++) k_spin.push_back(2);

    bool is_restricted = true;
    if(sys_data.options_map.scf_options.scf_type == "uhf") is_restricted = false;

    if(rank==0) {
        if(is_restricted) cout << endl << "Running Closed Shell CCSD(T) calculation" << endl;
        else cout << endl << "Running Open Shell CCSD(T) calculation" << endl;
    }

    bool seq_h3b=true;
    Index cache_size=32;
    LRUCache<Index,std::vector<T>> cache_s1t{cache_size};
    LRUCache<Index,std::vector<T>> cache_s1v{cache_size};
    LRUCache<Index,std::vector<T>> cache_d1t{cache_size*noab};
    LRUCache<Index,std::vector<T>> cache_d1v{cache_size*noab};
    LRUCache<Index,std::vector<T>> cache_d2t{cache_size*nvab};
    LRUCache<Index,std::vector<T>> cache_d2v{cache_size*nvab};

    if(rank==0 && seq_h3b) cout << "running seq h3b loop variant..." << endl;

    double ccsd_t_time = 0, total_t_time = 0;
    // cc_t1 = std::chrono::high_resolution_clock::now();

    std::tie(energy1,energy2,ccsd_t_time,total_t_time) = ccsd_t_fused_driver_new<T>(sys_data,ec,k_spin,MO,d_t1,d_t2,d_v2,
                                    p_evl_sorted,hf_energy+corr_energy,ccsd_options.ngpu,is_restricted,
                                    cache_s1t,cache_s1v,cache_d1t,
                                    cache_d1v,cache_d2t,cache_d2v,seq_h3b);

    // cc_t2 = std::chrono::high_resolution_clock::now();
    // auto ccsd_t_time = 
    //     std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();

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

    long double total_num_ops = 0;  
    //
    if (rank == 0)     
    {
        // std::cout << "--------------------------------------------------------------------" << std::endl;
        ccsd_t_fused_driver_calculator_ops<T>(sys_data,ec,k_spin,MO1,
                                    p_evl_sorted,hf_energy+corr_energy,ccsd_options.ngpu,is_restricted,
                                    total_num_ops, 
                                    seq_h3b);
        // std::cout << "--------------------------------------------------------------------" << std::endl;
    }

    ec.pg().barrier();

    auto nranks = ec.pg().size().value();

    auto print_profile_stats = [&](const std::string& timer_type, const double g_tval, const double tval_min, const double tval_max){
        const double tval = g_tval/nranks;
        std::cout.precision(3);
        std::cout << "   -> " << timer_type << ": " << tval << "s (" << tval*100.0/total_t_time << "%), (min,max) = (" << tval_min << "," << tval_max << ")" << std::endl;
    };

    auto comm_stats = [&](const std::string& timer_type, const double ctime){
        double g_getTime,g_min_getTime,g_max_getTime;
        MPI_Reduce(&ctime, &g_getTime,     1, MPI_DOUBLE, MPI_SUM, 0, ec.pg().comm());
        MPI_Reduce(&ctime, &g_min_getTime, 1, MPI_DOUBLE, MPI_MIN, 0, ec.pg().comm());
        MPI_Reduce(&ctime, &g_max_getTime, 1, MPI_DOUBLE, MPI_MAX, 0, ec.pg().comm());
        if(rank == 0) 
        print_profile_stats(timer_type, g_getTime, g_min_getTime, g_max_getTime);
        return g_getTime/nranks;        
    };

    if(rank == 0) {
      std::cout << std::endl << "------CCSD(T) Performance------" << std::endl;
      std::cout << "Total CCSD(T) Time: " << total_t_time << std::endl;
    }
    ccsd_t_time = comm_stats("CCSD(T) Avg. Work Time", ccsd_t_time);
    if(rank == 0) {
      std::cout << std::scientific << "   -> Total Number of Operations: " << total_num_ops << std::endl;
      std::cout << std::fixed << "   -> GFLOPS: " << total_num_ops / (total_t_time * 1e9) << std::endl;
      std::cout << std::fixed << "   -> Load imbalance: " << (1.0 - ccsd_t_time / total_t_time) << std::endl;
    }
    
    comm_stats("S1-T1 GetTime", ccsdt_s1_t1_GetTime);    
    comm_stats("S1-V2 GetTime", ccsdt_s1_v2_GetTime);
    comm_stats("D1-T2 GetTime", ccsdt_d1_t2_GetTime);
    comm_stats("D1-V2 GetTime", ccsdt_d1_v2_GetTime);
    comm_stats("D2-T2 GetTime", ccsdt_d2_t2_GetTime);
    comm_stats("D2-V2 GetTime", ccsdt_d2_v2_GetTime);

    double g_ccsd_t_data_per_rank;
    ccsd_t_data_per_rank = (ccsd_t_data_per_rank * 8.0) / (1024*1024.0*1024); //GB
    MPI_Reduce(&ccsd_t_data_per_rank, &g_ccsd_t_data_per_rank, 1, MPI_DOUBLE, MPI_SUM, 0, ec.pg().comm());
    if(rank == 0) 
        std::cout << "   -> Data Transfer (GB): " << g_ccsd_t_data_per_rank/nranks << std::endl;

    ec.pg().barrier();

   #if 1
    std::vector<Index> cvec_s1t;
    std::vector<Index> cvec_s1v;
    std::vector<Index> cvec_d1t;
    std::vector<Index> cvec_d1v;
    std::vector<Index> cvec_d2t;
    std::vector<Index> cvec_d2v;

    cache_s1t.gather_stats(cvec_s1t);
    cache_s1v.gather_stats(cvec_s1v);
    cache_d1t.gather_stats(cvec_d1t);
    cache_d1v.gather_stats(cvec_d1v);
    cache_d2t.gather_stats(cvec_d2t);
    cache_d2v.gather_stats(cvec_d2v);

    std::vector<Index> g_cvec_s1t(cvec_s1t.size());
    std::vector<Index> g_cvec_s1v(cvec_s1v.size());
    std::vector<Index> g_cvec_d1t(cvec_d1t.size());
    std::vector<Index> g_cvec_d1v(cvec_d1v.size());
    std::vector<Index> g_cvec_d2t(cvec_d2t.size());
    std::vector<Index> g_cvec_d2v(cvec_d2v.size());
    MPI_Reduce(&cvec_s1t[0], &g_cvec_s1t[0], cvec_s1t.size(), MPI_UINT32_T, MPI_SUM, 0, ec.pg().comm());
    MPI_Reduce(&cvec_s1v[0], &g_cvec_s1v[0], cvec_s1v.size(), MPI_UINT32_T, MPI_SUM, 0, ec.pg().comm());
    MPI_Reduce(&cvec_d1t[0], &g_cvec_d1t[0], cvec_d1t.size(), MPI_UINT32_T, MPI_SUM, 0, ec.pg().comm());           
    MPI_Reduce(&cvec_d1v[0], &g_cvec_d1v[0], cvec_d1v.size(), MPI_UINT32_T, MPI_SUM, 0, ec.pg().comm());           
    MPI_Reduce(&cvec_d2t[0], &g_cvec_d2t[0], cvec_d2t.size(), MPI_UINT32_T, MPI_SUM, 0, ec.pg().comm());           
    MPI_Reduce(&cvec_d2v[0], &g_cvec_d2v[0], cvec_d2v.size(), MPI_UINT32_T, MPI_SUM, 0, ec.pg().comm());           

    out_fp = sys_data.input_molecule+"."+sys_data.options_map.ccsd_options.basis;
    files_dir = out_fp+"_files/ccsd_t";
    if(seq_h3b) files_dir = out_fp+"_files/ccsd_t_seq_h3b";
    if(!fs::exists(files_dir)) fs::create_directories(files_dir);    
    std::string fp = files_dir+"/";

    auto print_stats = [&](std::ostream& os, std::vector<uint32_t>& vec){
      for (uint32_t i = 0; i < vec.size(); i++) {
        os << i << " : " << vec[i] << std::endl;
      }
    };

    if(rank == 0) {
      std::ofstream fp_s1t(fp+"s1t_stats");
      std::ofstream fp_s1v(fp+"s1v_stats");
      std::ofstream fp_d1t(fp+"d1t_stats");
      std::ofstream fp_d1v(fp+"d1v_stats");
      std::ofstream fp_d2t(fp+"d2t_stats");
      std::ofstream fp_d2v(fp+"d2v_stats");

      print_stats(fp_s1t,g_cvec_s1t);
      print_stats(fp_s1v,g_cvec_s1v);
      print_stats(fp_d1t,g_cvec_d1t);
      print_stats(fp_d1v,g_cvec_d1v);
      print_stats(fp_d2t,g_cvec_d2t);
      print_stats(fp_d2v,g_cvec_d2v);
    }
    #endif

    free_tensors(d_t1, d_t2, d_f1, d_v2);

    ec.flush_and_sync();
    // delete ec;

}
