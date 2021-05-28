#include "cd_ccsd_os_ann.hpp"

#include <filesystem>
namespace fs = std::filesystem;

void ccsd_driver();
std::string filename;

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

    ccsd_driver();

    tamm::finalize();

    return 0;
}

template<typename T>
Tensor2D read_ext_data_2d(std::string& movecsfile) {

  auto mfile_id = H5Fopen(movecsfile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  auto mdataset_id = H5Dopen(mfile_id, "fmo",  H5P_DEFAULT);

  // Read attributes - reduced dims
  std::vector<int> reduced_dims(2);
  auto attr_dataset = H5Dopen(mfile_id, "fmo_dims",  H5P_DEFAULT);
  H5Dread(attr_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, reduced_dims.data());
  
  /* Read the datasets. */
  Tensor2D mbuf(reduced_dims[0],reduced_dims[1]);
  mbuf.setZero();
  
  H5Dread(mdataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mbuf.data());

  H5Dclose(attr_dataset);
  H5Dclose(mdataset_id);
  H5Fclose(mfile_id);

  return mbuf;
}

template<typename T>
Tensor3D read_ext_data_3d(std::string& movecsfile) {

  auto mfile_id = H5Fopen(movecsfile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  auto mdataset_id = H5Dopen(mfile_id, "cholmo",  H5P_DEFAULT);

  // Read attributes - reduced dims
  std::vector<int> reduced_dims(3);
  auto attr_dataset = H5Dopen(mfile_id, "chol_dims",  H5P_DEFAULT);
  H5Dread(attr_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, reduced_dims.data());
  
  /* Read the datasets. */
  Tensor3D mbuf(reduced_dims[0],reduced_dims[1],reduced_dims[2]);
  mbuf.setZero();
  
  H5Dread(mdataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, mbuf.data());

  H5Dclose(attr_dataset);
  H5Dclose(mdataset_id);
  H5Fclose(mfile_id);

  return mbuf;
}

void ccsd_driver() {

    // std::cout << "Input file provided = " << filename << std::endl;

    using T = double;

    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    auto rank = ec.pg().rank();

    auto [sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, C_beta_AO, F_beta_AO, AO_opt, AO_tis,scf_conv]  
                    = hartree_fock_driver<T>(ec,filename);

    CCSDOptions& ccsd_options = sys_data.options_map.ccsd_options;
    debug = ccsd_options.debug;
    if(rank == 0) ccsd_options.print();

    if(rank==0) cout << endl << "#occupied, #virtual = " << sys_data.nocc << ", " << sys_data.nvir << endl;

    auto [MO,total_orbitals] = setupMOIS(sys_data);

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

    //deallocates F_AO, C_AO
    // auto [cholVpr,d_f1,chol_count, max_cvecs, CI] = cd_svd_ga_driver<T>
    //                     (sys_data, ec, MO, AO_opt, C_AO, F_AO, C_beta_AO, F_beta_AO, shells, shell_tile_map,
    //                             ccsd_restart, cholfile);
    // free_tensors(lcao);

    Tensor2D ext_fmo = read_ext_data_2d<T>(ccsd_options.ext_data_path);
    Tensor3D ext_chol = read_ext_data_3d<T>(ccsd_options.ext_data_path);

    TAMM_SIZE chol_count = 0;
    chol_count = ext_chol.dimensions()[2];
    
    sys_data.nmo = ext_chol.dimensions()[0];
    sys_data.nbf = sys_data.nmo/2;
    sys_data.nbf_orig = sys_data.nbf;
    sys_data.num_chol_vectors = chol_count;

    std::vector<int> occ_virt_sizes(4);
    {
        auto mfile_id = H5Fopen(ccsd_options.ext_data_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

        auto attr_dataset = H5Dopen(mfile_id, "occ_virt_sizes",  H5P_DEFAULT);
        H5Dread(attr_dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, occ_virt_sizes.data());
        
        H5Dclose(attr_dataset);
        H5Fclose(mfile_id);
    }

    sys_data.nocc = occ_virt_sizes[0]+occ_virt_sizes[1]; //occ_alpha+occ_beta
    sys_data.nvir = sys_data.nmo - sys_data.nocc;
    sys_data.n_occ_alpha = occ_virt_sizes[0];
    sys_data.n_occ_beta  = occ_virt_sizes[1];
    sys_data.n_vir_alpha = occ_virt_sizes[2];
    sys_data.n_vir_beta  = occ_virt_sizes[3];

    auto [MO,total_orbitals] = setupMOIS(sys_data);

    sys_data.print();

    TiledIndexSpace N = MO("all");

    Tensor<T> d_f1{{N,N},{1,1}};
    Tensor<T>::allocate(&ec,d_f1);
    Tensor<T> cholVpr;

    IndexSpace chol_is{range(0,chol_count)};
    TiledIndexSpace CI{chol_is,static_cast<tamm::Tile>(ccsd_options.itilesize)}; 

    cholVpr = {{N,N,CI},{SpinPosition::upper,SpinPosition::lower,SpinPosition::ignore}};
    Tensor<TensorType>::allocate(&ec, cholVpr);

    eigen_to_tamm_tensor(d_f1, ext_fmo);
    eigen_to_tamm_tensor(cholVpr, ext_chol);

    ext_fmo.resize(0,0);
    ext_chol.resize(0,0,0);

    Tensor<T>::deallocate(C_AO,F_AO);
    if(sys_data.scf_type == sys_data.SCFType::uhf) Tensor<T>::deallocate(C_beta_AO,F_beta_AO);

    //----------------------

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
    #ifdef USE_TALSH
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

    if(is_rhf)
        std::tie(residual, corr_energy) = cd_ccsd_cs_driver<T>(
            sys_data, ec, MO, CI, d_t1, d_t2, d_f1, 
            d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s, 
            p_evl_sorted, 
            cholVpr, ccsd_restart, files_prefix,
            computeTData);
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

    if(is_rhf) free_tensors(d_t1, d_t2, d_f1);
    else free_tensors(d_f1);

    Tensor<T> d_v2;
    if(computeTData) {
        d_v2 = setupV2<T>(ec,MO,CI,cholVpr,chol_count, ex_hw);
        if(ccsd_options.writev) {
          write_to_disk(d_v2,fullV2file,true);
          Tensor<T>::deallocate(d_v2);
        }
    }

    free_tensors(cholVpr);

    #ifdef USE_TALSH
    //talshStats();
    if(has_gpu) talsh_instance.shutdown();
    #endif

    if(computeTData) {
        if(!is_rhf) {
          dt1_full = d_t1;
          dt2_full = d_t2;
        }
        if(rank==0) {
          cout << endl << "Retile T1,T2,V2 ... " << endl;
        }

        auto [MO1,total_orbitals1] = setupMOIS(sys_data,true);
        TiledIndexSpace N1 = MO1("all");
        TiledIndexSpace O1 = MO1("occ");
        TiledIndexSpace V1 = MO1("virt");

        // Tensor<T> t_d_f1{{N1,N1},{1,1}};
        Tensor<T> t_d_t1{{V1,O1},{1,1}};
        Tensor<T> t_d_t2{{V1,V1,O1,O1},{2,2}};
        Tensor<T> t_d_v2{{N1,N1,N1,N1},{2,2}};
        Tensor<T>::allocate(&ec,t_d_t1,t_d_t2,t_d_v2);

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

        free_tensors(t_d_t1, t_d_t2, t_d_v2);
    }
    
    if(!is_rhf) free_tensors(d_t1, d_t2);

    ec.flush_and_sync();
    // delete ec;

}
