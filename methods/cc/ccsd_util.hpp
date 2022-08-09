#pragma once

#include <ctime>
#include "ccse_tensors.hpp"

// auto lambdar2 = [](const IndexVector& blockid, span<double> buf){
//     if((blockid[0] > blockid[1]) || (blockid[2] > blockid[3])) {
//         for(auto i = 0U; i < buf.size(); i++) buf[i] = 0; 
//     }
// };

template<typename T>
struct V2Tensors {
  Tensor<T> v2ijab; //hhpp
  Tensor<T> v2iajb; //hphp
  Tensor<T> v2ijka; //hhhp
  Tensor<T> v2ijkl; //hhhh
  Tensor<T> v2iabc; //hppp
  Tensor<T> v2abcd; //pppp

  std::string v2ijab_file,v2iajb_file,v2ijka_file,v2ijkl_file,v2iabc_file,v2abcd_file;

  void deallocate() {
    Tensor<T>::deallocate(v2ijab,v2iajb,v2ijka,v2ijkl,v2iabc,v2abcd);
  }

  void allocate(ExecutionContext& ec, const TiledIndexSpace& MO) {
    auto [h1,h2,h3,h4] = MO.labels<4>("occ");
    auto [p1,p2,p3,p4] = MO.labels<4>("virt");

    v2ijkl = Tensor<T>{{h1,h2,h3,h4},{2,2}};
    v2ijka = Tensor<T>{{h1,h2,h3,p1},{2,2}};
    v2iajb = Tensor<T>{{h1,p1,h2,p2},{2,2}};
    v2ijab = Tensor<T>{{h1,h2,p1,p2},{2,2}};
    v2iabc = Tensor<T>{{h1,p1,p2,p3},{2,2}};
    v2abcd = Tensor<T>{{p1,p2,p3,p4},{2,2}};
    Tensor<T>::allocate(&ec,v2ijab,v2iajb,v2ijka,v2ijkl,v2iabc,v2abcd);
  }

  void set_file_prefix(const std::string& fprefix){
    v2ijab_file = fprefix+".v2ijab";
    v2iajb_file = fprefix+".v2iajb";
    v2ijka_file = fprefix+".v2ijka";
    v2ijkl_file = fprefix+".v2ijkl";
    v2iabc_file = fprefix+".v2iabc";
    v2abcd_file = fprefix+".v2abcd";
  }

  void write_to_disk(const std::string& fprefix){
    set_file_prefix(fprefix);
    //TODO: Assume all on same ec for now
    ExecutionContext& ec = get_ec(v2ijab());
    tamm::write_to_disk_group<T>(ec,{v2ijab,v2iajb,v2ijka,v2ijkl,v2iabc,v2abcd},
    {v2ijab_file,v2iajb_file,v2ijka_file,v2ijkl_file,v2iabc_file,v2abcd_file});
  }

  void read_from_disk(const std::string& fprefix){
    set_file_prefix(fprefix);
    ExecutionContext& ec = get_ec(v2ijab());
    tamm::read_from_disk_group<T>(ec,{v2ijab,v2iajb,v2ijka,v2ijkl,v2iabc,v2abcd},
    {v2ijab_file,v2iajb_file,v2ijka_file,v2ijkl_file,v2iabc_file,v2abcd_file});
  }

  bool exist_on_disk(const std::string& fprefix) {
    set_file_prefix(fprefix);
    return ( fs::exists(v2ijab_file) && fs::exists(v2iajb_file) &&
             fs::exists(v2ijka_file) && fs::exists(v2ijkl_file) &&
             fs::exists(v2iabc_file) && fs::exists(v2abcd_file) );
  }
};

template<typename T>
void setup_full_t1t2(ExecutionContext& ec, const TiledIndexSpace& MO,
  Tensor<T>& dt1_full, Tensor<T>& dt2_full) {

  TiledIndexSpace O = MO("occ");
  TiledIndexSpace V = MO("virt");

  dt1_full = Tensor<T>{{V,O},{1,1}};
  dt2_full = Tensor<T>{{V,V,O,O},{2,2}};

  Tensor<TensorType>::allocate(&ec,dt1_full,dt2_full);
  // (dt1_full() = 0)
  // (dt2_full() = 0)
}

template<typename TensorType>
void update_r2(ExecutionContext& ec, 
              LabeledTensor<TensorType> ltensor) {
    Tensor<TensorType> tensor = ltensor.tensor();

    auto lambda = [&](const IndexVector& bid) {
        const IndexVector blockid   = internal::translate_blockid(bid, ltensor);
        if((blockid[0] > blockid[1]) || (blockid[2] > blockid[3])) {
          const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
          std::vector<TensorType> dbuf(dsize);
          tensor.get(blockid, dbuf);
          // func(blockid, dbuf);
          for(auto i = 0U; i < dsize; i++) dbuf[i] = 0; 
          tensor.put(blockid, dbuf);
        }
    };
    block_for(ec, ltensor, lambda);
}

template<typename TensorType>
void update_gamma2(ExecutionContext& ec, 
              LabeledTensor<TensorType> ltensor) {
    Tensor<TensorType> tensor = ltensor.tensor();
    auto tis = tensor.tiled_index_spaces();

    auto lambda = [&](const IndexVector& bid) {
        const IndexVector blockid   = internal::translate_blockid(bid, ltensor);
        if(  (tis[0].spin(blockid[0]) != tis[2].spin(blockid[2])) 
          || (tis[1].spin(blockid[1]) != tis[3].spin(blockid[3])) ) {
          const tamm::TAMM_SIZE dsize = tensor.block_size(blockid);
          std::vector<TensorType> dbuf(dsize);
          tensor.get(blockid, dbuf);
          // func(blockid, dbuf);
          for(auto i = 0U; i < dsize; i++) dbuf[i] = 0; 
          tensor.put(blockid, dbuf);
        }
    };
    block_for(ec, ltensor, lambda);
}


template<typename TensorType>
void init_diagonal(ExecutionContext& ec,
                    LabeledTensor<TensorType> ltensor) {
    Tensor<TensorType> tensor = ltensor.tensor();
    // Defined only for NxN tensors
    EXPECTS(tensor.num_modes() == 2);

    auto lambda = [&](const IndexVector& bid) {
        const IndexVector blockid   = internal::translate_blockid(bid, ltensor);
        if(blockid[0] == blockid[1]) {
          const TAMM_SIZE size = tensor.block_size(blockid);
            std::vector<TensorType> buf(size);
            tensor.get(blockid, buf);
            auto block_dims   = tensor.block_dims(blockid);
            auto block_offset = tensor.block_offsets(blockid);
            auto dim          = block_dims[0];
            auto offset       = block_offset[0];
            size_t i          = 0;
            for(auto p = offset; p < offset + dim; p++, i++)
               buf[i * dim + i] = 1.0;
          tensor.put(blockid, buf);
        }
    };
    block_for(ec, ltensor, lambda);
}

std::string ccsd_test( int argc, char* argv[] )
{

    if(argc<2){
        std::cout << "Please provide an input file!" << std::endl;
        exit(0);
    }

    auto filename = std::string(argv[1]);
    std::ifstream testinput(filename); 
    if(!testinput){
        std::cout << "Input file provided [" << filename << "] does not exist!" << std::endl;
        exit(0);
    }

    return filename;
}

void iteration_print(SystemData& sys_data, const ProcGroup& pg, int iter, double residual, 
                      double energy, double time, string cmethod="CCSD") {
  if(pg.rank() == 0) {
    std::cout.width(6); std::cout << std::right << iter+1 << "  ";
    std::cout << std::setprecision(13) << residual << "  ";
    std::cout << std::fixed << std::setprecision(13) << energy << " ";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::string(4, ' ') << "0.0";
    std::cout << std::string(5, ' ') << time;
    std::cout << std::string(5, ' ') << "0.0" << std::endl;

    sys_data.results["output"][cmethod]["iter"][std::to_string(iter+1)] = { {"residual", residual}, {"correlation", energy} };
    sys_data.results["output"][cmethod]["iter"][std::to_string(iter+1)]["performance"] = { {"total_time", time} };
  }
}

void iteration_print_lambda(const ProcGroup& pg, int iter, double residual, double time) {
  if(pg.rank() == 0) {
    std::cout.width(6); std::cout << std::right << iter+1 << "  ";
    std::cout << std::setprecision(13) << residual << "  ";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::string(8, ' ') << "0.0";
    std::cout << std::string(5, ' ') << time << std::endl;
  }
}

/**
 *
 * @tparam T
 * @param MO
 * @param p_evl_sorted
 * @return pair of residual and energy
 */
template<typename T>
std::pair<double,double> rest(ExecutionContext& ec,
                              const TiledIndexSpace& MO,
                              Tensor<T>& d_r1,
                              Tensor<T>& d_r2,
                              Tensor<T>& d_t1,
                              Tensor<T>& d_t2,
                              Tensor<T>& de,
                              Tensor<T>& d_r1_residual,
                              Tensor<T>& d_r2_residual,                              
                              std::vector<T>& p_evl_sorted, T zshiftl, 
                              const TAMM_SIZE& noa,
                              const TAMM_SIZE& nob, bool transpose=false) {

    T residual, energy;
    Scheduler sch{ec};
    // Tensor<T> d_r1_residual{}, d_r2_residual{};
    // Tensor<T>::allocate(&ec,d_r1_residual, d_r2_residual);
    sch
      (d_r1_residual() = d_r1()  * d_r1())
      (d_r2_residual() = d_r2()  * d_r2())
      .execute();

      auto l0 = [&]() {
        T r1 = get_scalar(d_r1_residual);
        T r2 = get_scalar(d_r2_residual);
        r1 = 0.5*std::sqrt(r1);
        r2 = 0.5*std::sqrt(r2);
        energy = get_scalar(de);
        residual = std::max(r1,r2);
      };

      auto l1 =  [&]() {
        jacobi(ec, d_r1, d_t1, -1.0 * zshiftl, transpose, p_evl_sorted,noa,nob);
      };
      auto l2 = [&]() {
        jacobi(ec, d_r2, d_t2, -2.0 * zshiftl, transpose, p_evl_sorted,noa,nob);
      };

      l0();
      l1();
      l2();

      // Tensor<T>::deallocate(d_r1_residual, d_r2_residual);
      
    return {residual, energy};
}

template<typename T>
std::pair<double,double> rest_cs(ExecutionContext& ec,
                              const TiledIndexSpace& MO,
                              Tensor<T>& d_r1,
                              Tensor<T>& d_r2,
                              Tensor<T>& d_t1,
                              Tensor<T>& d_t2,
                              Tensor<T>& de,
                              Tensor<T>& d_r1_residual,
                              Tensor<T>& d_r2_residual,
                              std::vector<T>& p_evl_sorted, T zshiftl, 
                              const TAMM_SIZE& noa,
                              const TAMM_SIZE& nva, bool transpose=false,
			       const bool not_spin_orbital=false) {

    T residual, energy;
    Scheduler sch{ec};
    // Tensor<T> d_r1_residual{}, d_r2_residual{};
    // Tensor<T>::allocate(&ec,d_r1_residual, d_r2_residual);
    sch
      (d_r1_residual() = d_r1()  * d_r1())
      (d_r2_residual() = d_r2()  * d_r2())
      .execute();

      auto l0 = [&]() {
        T r1 = get_scalar(d_r1_residual);
        T r2 = get_scalar(d_r2_residual);
        r1 = 0.5*std::sqrt(r1);
        r2 = 0.5*std::sqrt(r2);
        energy = get_scalar(de);
        residual = std::max(r1,r2);
      };

      auto l1 =  [&]() {
        jacobi_cs(ec, d_r1, d_t1, -1.0 * zshiftl, transpose, p_evl_sorted,noa,nva,not_spin_orbital);
      };
      auto l2 = [&]() {
        jacobi_cs(ec, d_r2, d_t2, -2.0 * zshiftl, transpose, p_evl_sorted,noa,nva,not_spin_orbital);
      };

      l0();
      l1();
      l2();

      // Tensor<T>::deallocate(d_r1_residual, d_r2_residual);
      
    return {residual, energy};
}

void print_ccsd_header(const bool do_print) {
  if(do_print) {
    std::cout << std::endl << std::endl;
    std::cout << " CCSD iterations" << std::endl;
    std::cout << std::string(66, '-') << std::endl;
    std::cout <<
        " Iter          Residuum       Correlation     Cpu    Wall    V2*C2"
              << std::endl;
    std::cout << std::string(66, '-') << std::endl;
  }
}

template<typename T>
std::tuple<std::vector<T>,Tensor<T>,Tensor<T>,Tensor<T>,Tensor<T>,
std::vector<Tensor<T>>,std::vector<Tensor<T>>,std::vector<Tensor<T>>,std::vector<Tensor<T>>>
 setupTensors(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis, bool ccsd_restart=false) {

    auto rank = ec.pg().rank();

    TiledIndexSpace O = MO("occ");
    TiledIndexSpace V = MO("virt");
 
    std::vector<T> p_evl_sorted = tamm::diagonal(d_f1);

    // auto lambda2 = [&](const IndexVector& blockid, span<T> buf){
    //     if(blockid[0] != blockid[1]) {
    //         for(auto i = 0U; i < buf.size(); i++) buf[i] = 0; 
    //     }
    // };

    // update_tensor(d_f1(),lambda2);
   
  std::vector<Tensor<T>> d_r1s, d_r2s, d_t1s, d_t2s;
  Tensor<T> d_r1{{V,O},{1,1}};
  Tensor<T> d_r2{{V,V,O,O},{2,2}};

  if(!ccsd_restart){
    for(decltype(ndiis) i=0; i<ndiis; i++) {
      d_r1s.push_back(Tensor<T>{{V,O},{1,1}});
      d_r2s.push_back(Tensor<T>{{V,V,O,O},{2,2}});
      d_t1s.push_back(Tensor<T>{{V,O},{1,1}});
      d_t2s.push_back(Tensor<T>{{V,V,O,O},{2,2}});
      Tensor<T>::allocate(&ec,d_r1s[i], d_r2s[i], d_t1s[i], d_t2s[i]);
    }
    Tensor<T>::allocate(&ec,d_r1,d_r2);
  }

  Tensor<T> d_t1{{V,O},{1,1}};
  Tensor<T> d_t2{{V,V,O,O},{2,2}};

  Tensor<T>::allocate(&ec,d_t1,d_t2);

  Scheduler{ec}   
  (d_t1() = 0)
  (d_t2() = 0)
  .execute();
  
  return std::make_tuple(p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s);
}

template<typename T>
std::tuple<std::vector<T>, Tensor<T>,Tensor<T>,Tensor<T>,Tensor<T>,
std::vector<Tensor<T>>,std::vector<Tensor<T>>,std::vector<Tensor<T>>,std::vector<Tensor<T>>>
 setupTensors_cs(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis, bool ccsd_restart=false) {

    auto rank = ec.pg().rank();

    const TiledIndexSpace &O = MO("occ");
    const TiledIndexSpace &V = MO("virt");
    
    const int otiles = O.num_tiles();
    const int vtiles = V.num_tiles();
    const int oatiles = MO("occ_alpha").num_tiles();
    const int obtiles = MO("occ_beta").num_tiles();
    const int vatiles = MO("virt_alpha").num_tiles();
    const int vbtiles = MO("virt_beta").num_tiles();

    TiledIndexSpace o_alpha,v_alpha,o_beta,v_beta;
    o_alpha = {MO("occ"), range(oatiles)};
    v_alpha = {MO("virt"), range(vatiles)};
    o_beta = {MO("occ"), range(obtiles,otiles)};
    v_beta = {MO("virt"), range(vbtiles,vtiles)};
 
    std::vector<T> p_evl_sorted = tamm::diagonal(d_f1);

    // auto lambda2 = [&](const IndexVector& blockid, span<T> buf){
    //     if(blockid[0] != blockid[1]) {
    //         for(auto i = 0U; i < buf.size(); i++) buf[i] = 0; 
    //     }
    // };

    // update_tensor(d_f1(),lambda2);

  std::vector<Tensor<T>> d_r1s, d_r2s, d_t1s, d_t2s;
  Tensor<T> d_r1{{v_alpha,o_alpha},{1,1}};
  Tensor<T> d_r2{{v_alpha,v_beta,o_alpha,o_beta},{2,2}};

  if(!ccsd_restart){
    for(decltype(ndiis) i=0; i<ndiis; i++) {
      d_r1s.push_back(Tensor<T>{{v_alpha,o_alpha},{1,1}});
      d_r2s.push_back(Tensor<T>{{v_alpha,v_beta,o_alpha,o_beta},{2,2}});
      d_t1s.push_back(Tensor<T>{{v_alpha,o_alpha},{1,1}});
      d_t2s.push_back(Tensor<T>{{v_alpha,v_beta,o_alpha,o_beta},{2,2}});
      Tensor<T>::allocate(&ec,d_r1s[i], d_r2s[i], d_t1s[i], d_t2s[i]);
    }
    Tensor<T>::allocate(&ec,d_r1,d_r2);
  }

  Tensor<T> d_t1{{v_alpha,o_alpha},{1,1}};
  Tensor<T> d_t2{{v_alpha,v_beta,o_alpha,o_beta},{2,2}};

  Tensor<T>::allocate(&ec,d_t1,d_t2);

  Scheduler{ec}   
  (d_t1() = 0)
  (d_t2() = 0)
  .execute();
  
  return std::make_tuple(p_evl_sorted,d_t1,d_t2,d_r1,d_r2, d_r1s, d_r2s, d_t1s, d_t2s);
}

template<typename T>
std::tuple<SystemData, double, 
  libint2::BasisSet, std::vector<size_t>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, TiledIndexSpace, TiledIndexSpace,bool> 
    hartree_fock_driver(ExecutionContext &ec, const string filename) {

    auto rank = ec.pg().rank();

    auto current_time = std::chrono::system_clock::now();
    auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
    auto cur_local_time = localtime(&current_time_t);

    if(rank == 0) cout << endl << "Date: " << std::put_time(cur_local_time, "%c") << endl << endl;

    double hf_energy{0.0};
    libint2::BasisSet shells;
    Tensor<T> C_AO, C_beta_AO;
    Tensor<T> F_AO, F_beta_AO;
    TiledIndexSpace tAO; //Fixed Tilesize AO
    TiledIndexSpace tAOt; //original AO TIS
    std::vector<size_t> shell_tile_map;
    bool scf_conv;

    // read geometry from a json file
    json jinput;
    check_json(filename);
    auto is = std::ifstream(filename);
    OptionsMap options_map;
    std::tie(options_map, jinput) = parse_input(is);
    if(options_map.options.output_file_prefix.empty()) 
      options_map.options.output_file_prefix = getfilename(filename);
    
    SystemData sys_data{options_map, options_map.scf_options.scf_type};

    auto hf_t1 = std::chrono::high_resolution_clock::now();

    std::tie(sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, C_beta_AO, F_beta_AO, tAO, tAOt, scf_conv) = hartree_fock(ec, filename, options_map);
    sys_data.input_molecule = getfilename(filename);
    sys_data.output_file_prefix = options_map.options.output_file_prefix;

    auto hf_t2 = std::chrono::high_resolution_clock::now();

    double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << std::endl << "Time taken for Hartree-Fock: " << hf_time << " secs" << std::endl;

    return std::make_tuple(sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, C_beta_AO, F_beta_AO, tAO, tAOt, scf_conv);
}


void ccsd_stats(ExecutionContext& ec, double hf_energy,double residual,double energy,double thresh){

    auto rank = ec.pg().rank();
    bool ccsd_conv = residual < thresh;
    if(rank == 0) {
      std::cout << std::string(66, '-') << std::endl;
      if(ccsd_conv) {
          std::cout << " Iterations converged" << std::endl;
          std::cout.precision(15);
          std::cout << " CCSD correlation energy / hartree ="
                    << std::setw(26) << std::right << energy
                    << std::endl;
          std::cout << " CCSD total energy / hartree       ="
                    << std::setw(26) << std::right
                    << energy + hf_energy << std::endl;
      }    
    }
    if(!ccsd_conv){
      ec.pg().barrier();
      tamm_terminate("ERROR: CCSD calculation does not converge!");
    }

}

// template<typename T> 
// void freeTensors(size_t ndiis, Tensor<T>& d_r1, Tensor<T>& d_r2, Tensor<T>& d_t1, Tensor<T>& d_t2,
//                    Tensor<T>& d_f1, std::vector<Tensor<T>>& d_r1s, 
//                    std::vector<Tensor<T>>& d_r2s, std::vector<Tensor<T>>& d_t1s, 
//                    std::vector<Tensor<T>>& d_t2s) {

//   for(auto i=0; i<ndiis; i++)
//     Tensor<T>::deallocate(d_r1s[i], d_r2s[i], d_t1s[i], d_t2s[i]);
//   d_r1s.clear(); d_r2s.clear();
//   d_t1s.clear(); d_t2s.clear();
//   Tensor<T>::deallocate(d_r1, d_r2, d_t1, d_t2, d_f1);//, d_v2);
// }

auto sum_tensor_sizes = [](auto&&... t) {
    return ( ( compute_tensor_size(t) + ...) * 8 ) / (1024*1024*1024.0);
};

auto free_vec_tensors = [](auto&&... vecx) {
    (std::for_each(vecx.begin(), vecx.end(), [](auto& t) { t.deallocate(); }),
      ...);
};

auto free_tensors = [](auto&&... t) {
    ( (t.deallocate()), ...);
};


template<typename T>
std::tuple<Tensor<T>,Tensor<T>,Tensor<T>,Tensor<T>,std::vector<Tensor<T>>,std::vector<Tensor<T>>,
std::vector<Tensor<T>>,std::vector<Tensor<T>>> 
setupLambdaTensors(ExecutionContext& ec, TiledIndexSpace& MO, size_t ndiis) {

    TiledIndexSpace O = MO("occ");
    TiledIndexSpace V = MO("virt");
    
    auto rank = ec.pg().rank();

    Tensor<T> d_r1{{O,V},{1,1}};
    Tensor<T> d_r2{{O,O,V,V},{2,2}};
    Tensor<T> d_y1{{O,V},{1,1}};
    Tensor<T> d_y2{{O,O,V,V},{2,2}};

    Tensor<T>::allocate(&ec,d_r1, d_r2,d_y1,d_y2);

  Scheduler{ec}
      (d_y1() = 0)
      (d_y2() = 0)      
      (d_r1() = 0)
      (d_r2() = 0)
    .execute();

  if(rank == 0) {
    std::cout << std::endl << std::endl;
    std::cout << " Lambda CCSD iterations" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    std::cout <<
        " Iter          Residuum          Cpu    Wall"
              << std::endl;
    std::cout << std::string(45, '-') << std::endl;
  }

  std::vector<Tensor<T>> d_r1s,d_r2s,d_y1s, d_y2s;

  for(size_t i=0; i<ndiis; i++) {
    d_r1s.push_back(Tensor<T>{{O,V},{1,1}});
    d_r2s.push_back(Tensor<T>{{O,O,V,V},{2,2}});

    d_y1s.push_back(Tensor<T>{{O,V},{1,1}});
    d_y2s.push_back(Tensor<T>{{O,O,V,V},{2,2}});
    Tensor<T>::allocate(&ec,d_r1s[i],d_r2s[i],d_y1s[i], d_y2s[i]);
  }

    return std::make_tuple(d_r1,d_r2,d_y1,d_y2,d_r1s,d_r2s,d_y1s,d_y2s);

}


template<typename T>
V2Tensors<T> setupV2Tensors(ExecutionContext& ec, Tensor<T> cholVpr, ExecutionHW ex_hw = ExecutionHW::CPU) {
  TiledIndexSpace MO    = cholVpr.tiled_index_spaces()[0]; // MO
  TiledIndexSpace CI    = cholVpr.tiled_index_spaces()[2]; // CI
  auto [cind]           = CI.labels<1>("all");
  auto [h1, h2, h3, h4] = MO.labels<4>("occ");
  auto [p1, p2, p3, p4] = MO.labels<4>("virt");

  V2Tensors<T> v2tensors;
  v2tensors.allocate(ec, MO);

  // clang-format off
  Scheduler{ec}
  ( v2tensors.v2ijkl(h1,h2,h3,h4)      =   1.0 * cholVpr(h1,h3,cind) * cholVpr(h2,h4,cind) )
  ( v2tensors.v2ijkl(h1,h2,h3,h4)     +=  -1.0 * cholVpr(h1,h4,cind) * cholVpr(h2,h3,cind) )
  ( v2tensors.v2ijka(h1,h2,h3,p1)      =   1.0 * cholVpr(h1,h3,cind) * cholVpr(h2,p1,cind) )
  ( v2tensors.v2ijka(h1,h2,h3,p1)     +=  -1.0 * cholVpr(h2,h3,cind) * cholVpr(h1,p1,cind) )
  ( v2tensors.v2iajb(h1,p1,h2,p2)      =   1.0 * cholVpr(h1,h2,cind) * cholVpr(p1,p2,cind) )
  ( v2tensors.v2iajb(h1,p1,h2,p2)     +=  -1.0 * cholVpr(h1,p2,cind) * cholVpr(h2,p1,cind) )
  ( v2tensors.v2ijab(h1,h2,p1,p2)      =   1.0 * cholVpr(h1,p1,cind) * cholVpr(h2,p2,cind) )
  ( v2tensors.v2ijab(h1,h2,p1,p2)     +=  -1.0 * cholVpr(h1,p2,cind) * cholVpr(h2,p1,cind) )
  ( v2tensors.v2iabc(h1,p1,p2,p3)      =   1.0 * cholVpr(h1,p2,cind) * cholVpr(p1,p3,cind) )
  ( v2tensors.v2iabc(h1,p1,p2,p3)     +=  -1.0 * cholVpr(h1,p3,cind) * cholVpr(p1,p2,cind) )
  ( v2tensors.v2abcd(p1,p2,p3,p4)      =   1.0 * cholVpr(p1,p3,cind) * cholVpr(p2,p4,cind) )
  ( v2tensors.v2abcd(p1,p2,p3,p4)     +=  -1.0 * cholVpr(p1,p4,cind) * cholVpr(p2,p3,cind) )
  .execute(ex_hw);
  // clang-format on

  return v2tensors;
}

template<typename T>
Tensor<T> setupV2(ExecutionContext& ec, TiledIndexSpace& MO, TiledIndexSpace& CI,
                  Tensor<T> cholVpr, const tamm::Tile chol_count, 
                  ExecutionHW hw = ExecutionHW::CPU) {

    auto rank = ec.pg().rank();

    TiledIndexSpace N = MO("all");
    auto [cindex] = CI.labels<1>("all");
    auto [p,q,r,s] = MO.labels<4>("all");

    //Spin here is defined as spin(p)=spin(r) and spin(q)=spin(s) which is not currently not supported by TAMM.
    // Tensor<T> d_a2{{N,N,N,N},{2,2}};
    //For V2, spin(p)+spin(q) == spin(r)+spin(s)
    Tensor<T> d_v2{{N,N,N,N},{2,2}};
    Tensor<T>::allocate(&ec,d_v2);

    auto cc_t1 = std::chrono::high_resolution_clock::now();

    Scheduler{ec}(d_v2(p, q, r, s)  = cholVpr(p, r, cindex) * cholVpr(q, s, cindex))
                 (d_v2(p, q, r, s) += -1.0 * cholVpr(p, s, cindex) * cholVpr(q, r, cindex))
                 .execute(hw);

    auto cc_t2 = std::chrono::high_resolution_clock::now();
    double v2_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(rank == 0) std::cout << std::endl << "Time to reconstruct V2: " << v2_time << " secs" << std::endl;

    // Tensor<T>::deallocate(d_a2);
    return d_v2;

}

template<typename T> 
std::tuple<Tensor<T>,Tensor<T>,Tensor<T>,TAMM_SIZE, tamm::Tile, TiledIndexSpace>  
  cd_svd_ga_driver(SystemData& sys_data, ExecutionContext& ec, TiledIndexSpace& MO, TiledIndexSpace& AO,
  Tensor<T> C_AO, Tensor<T> F_AO, Tensor<T> C_beta_AO, Tensor<T> F_beta_AO, 
  libint2::BasisSet& shells, std::vector<size_t>& shell_tile_map, bool readv2=false, 
  std::string cholfile="", bool is_dlpno=false) {

    CDOptions cd_options = sys_data.options_map.cd_options;
    auto diagtol = cd_options.diagtol; // tolerance for the max. diagonal
    cd_options.max_cvecs_factor = 2 * std::abs(std::log10(diagtol));
    //TODO
    tamm::Tile max_cvecs = cd_options.max_cvecs_factor * sys_data.nbf;


    std::cout << std::defaultfloat;
    auto rank = ec.pg().rank();
    if(rank==0) cd_options.print();

    TiledIndexSpace N = MO("all");

    Tensor<T> d_f1{{N,N},{1,1}};
    Tensor<T> lcao{AO, N};
    Tensor<T>::allocate(&ec,d_f1,lcao);

    auto hf_t1        = std::chrono::high_resolution_clock::now();
    TAMM_SIZE chol_count = 0;

    //std::tie(V2) = 
    Tensor<T> cholVpr;

    auto itile_size = sys_data.options_map.ccsd_options.itilesize;

    sys_data.n_frozen_core    = sys_data.options_map.ccsd_options.freeze_core;
    sys_data.n_frozen_virtual = sys_data.options_map.ccsd_options.freeze_virtual;
    bool do_freeze = sys_data.n_frozen_core > 0 || sys_data.n_frozen_virtual > 0;

    std::string out_fp = sys_data.output_file_prefix+"."+sys_data.options_map.ccsd_options.basis;
    std::string files_dir = out_fp+"_files/"+sys_data.options_map.scf_options.scf_type;
    std::string lcaofile = files_dir+"/"+out_fp+".lcao";

    if(!readv2) {
      two_index_transform(sys_data, ec, C_AO, F_AO, C_beta_AO,F_beta_AO, d_f1, shells, lcao, is_dlpno);
      if(!is_dlpno) cholVpr = cd_svd_ga(sys_data, ec, MO, AO, chol_count, max_cvecs, shells, lcao);
      write_to_disk<TensorType>(lcao,lcaofile);
    }
    else{
      std::ifstream in(cholfile, std::ios::in);
      int rstatus = 0;
      if(in.is_open()) rstatus = 1;
      if(rstatus == 1) in >> chol_count; 
      else tamm_terminate("Error reading " + cholfile);

      if(rank==0) cout << "Number of cholesky vectors to be read = " << chol_count << endl;

      if(!is_dlpno) update_sysdata(sys_data, MO);

      IndexSpace chol_is{range(0,chol_count)};
      TiledIndexSpace CI{chol_is,static_cast<tamm::Tile>(itile_size)}; 

      TiledIndexSpace N = MO("all");
      cholVpr = {{N,N,CI},{SpinPosition::upper,SpinPosition::lower,SpinPosition::ignore}};
      if(!is_dlpno) Tensor<TensorType>::allocate(&ec, cholVpr);
      // Scheduler{ec}(cholVpr()=0).execute();
      read_from_disk(lcao,lcaofile);
    }

    auto hf_t2        = std::chrono::high_resolution_clock::now();
    double cd_svd_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    if(rank == 0) std::cout << std::endl << "Total Time taken for CD (+SVD): " << cd_svd_time
              << " secs" << std::endl;

    Tensor<T>::deallocate(C_AO,F_AO);
    if(sys_data.is_unrestricted) Tensor<T>::deallocate(C_beta_AO,F_beta_AO);

    IndexSpace chol_is{range(0,chol_count)};
    TiledIndexSpace CI{chol_is,static_cast<tamm::Tile>(itile_size)}; 

    sys_data.num_chol_vectors = chol_count;
    sys_data.results["output"]["CD"]["n_cholesky_vectors"] = chol_count;

    if(rank == 0) sys_data.print();

    if(do_freeze) {
      TiledIndexSpace N_eff = MO("all");
      Tensor<T> d_f1_new{{N_eff,N_eff},{1,1}};      
      Tensor<T>::allocate(&ec,d_f1_new);
      if(rank==0) {
        Matrix f1_eig     = tamm_to_eigen_matrix(d_f1);
        Matrix f1_new_eig = reshape_mo_matrix(sys_data,f1_eig);
        eigen_to_tamm_tensor(d_f1_new,f1_new_eig);
        f1_new_eig.resize(0,0);
      }
      Tensor<T>::deallocate(d_f1);
      d_f1 = d_f1_new;
    }

    if(!readv2 && sys_data.options_map.scf_options.print_mos.first) {
      Scheduler sch{ec};
      std::string hcorefile = files_dir+"/scf/"+out_fp+".hcore";
      Tensor<T> hcore{AO, AO};
      Tensor<T> hcore_mo{MO, MO};
      Tensor<T>::allocate(&ec,hcore,hcore_mo);
      read_from_disk(hcore,hcorefile);

      auto [mu,nu]   = AO.labels<2>("all");
      auto [mo1,mo2] = MO.labels<2>("all");
      
      Tensor<T> tmp{MO,AO};
      sch.allocate(tmp)
          (tmp(mo1,nu) = lcao(mu,mo1) * hcore(mu,nu))
          (hcore_mo(mo1,mo2) = tmp(mo1,nu) * lcao(nu,mo2))
          .deallocate(tmp,hcore).execute();

      ExecutionContext ec_dense{ec.pg(), DistributionKind::dense, MemoryManagerKind::ga};
      std::string mop_dir = files_dir+"/print_mos/";
      std::string mofprefix = mop_dir + out_fp;
      if(!fs::exists(mop_dir)) fs::create_directories(mop_dir);

      Tensor<T> d_v2 = setupV2<T>(ec,MO,CI,cholVpr,chol_count);
      Tensor<T> d_f1_dense  = to_dense_tensor(ec_dense, d_f1);
      Tensor<T> lcao_dense  = to_dense_tensor(ec_dense, lcao);
      Tensor<T> d_v2_dense  = to_dense_tensor(ec_dense, d_v2);
      Tensor<T> hcore_dense = to_dense_tensor(ec_dense, hcore_mo);

      Tensor<T>::deallocate(hcore_mo,d_v2);
      
      print_dense_tensor(d_v2_dense,  mofprefix+".v2_mo");
      print_dense_tensor(lcao_dense,  mofprefix+".ao2mo");
      print_dense_tensor(d_f1_dense,  mofprefix+".fock_mo");
      print_dense_tensor(hcore_dense, mofprefix+".hcore_mo");

      Tensor<T>::deallocate(hcore_dense,d_f1_dense,lcao_dense,d_v2_dense);
    }


    return std::make_tuple(cholVpr, d_f1, lcao, chol_count, max_cvecs, CI);
}
