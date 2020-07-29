#ifndef TESTS_CCSD_UTIL_HPP_
#define TESTS_CCSD_UTIL_HPP_

// #include "cd_svd.hpp"
#include "cd_svd/cd_svd_ga.hpp"
#include "macdecls.h"
#include "ga-mpi.h"


using namespace tamm;
using Tensor2D   = Eigen::Tensor<double, 2, Eigen::RowMajor>;
using Tensor3D   = Eigen::Tensor<double, 3, Eigen::RowMajor>;
using Tensor4D   = Eigen::Tensor<double, 4, Eigen::RowMajor>;


  // auto lambdar2 = [](const IndexVector& blockid, span<double> buf){
  //     if((blockid[0] > blockid[1]) || (blockid[2] > blockid[3])) {
  //         for(auto i = 0U; i < buf.size(); i++) buf[i] = 0; 
  //     }
  // };

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

void iteration_print(const ProcGroup& pg, int iter, double residual, double energy, double time) {
  if(pg.rank() == 0) {
    std::cout.width(6); std::cout << std::right << iter+1 << "  ";
    std::cout << std::setprecision(13) << residual << "  ";
    std::cout << std::fixed << std::setprecision(13) << energy << " ";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::string(4, ' ') << "0.0";
    std::cout << std::string(5, ' ') << time;
    std::cout << std::string(5, ' ') << "0.0" << std::endl;
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
                              const TAMM_SIZE& nva, bool transpose=false) {

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
        jacobi_cs(ec, d_r1, d_t1, -1.0 * zshiftl, transpose, p_evl_sorted,noa,nva);
      };
      auto l2 = [&]() {
        jacobi_cs(ec, d_r2, d_t2, -2.0 * zshiftl, transpose, p_evl_sorted,noa,nva);
      };

      l0();
      l1();
      l2();

      // Tensor<T>::deallocate(d_r1_residual, d_r2_residual);
      
    return {residual, energy};
}

std::tuple<TiledIndexSpace,TAMM_SIZE> setupMOIS(SystemData sys_data, bool triples=false) {
  
    // TAMM_SIZE nao = sys_data.nbf;
    TAMM_SIZE n_occ_alpha = sys_data.n_occ_alpha;
    TAMM_SIZE n_occ_beta = sys_data.n_occ_beta;
    TAMM_SIZE freeze_core = sys_data.n_frozen_core;
    TAMM_SIZE freeze_virtual = sys_data.n_frozen_virtual;

    Tile tce_tile = sys_data.options_map.ccsd_options.tilesize;
    if(!triples) {
      if ((tce_tile < static_cast<Tile>(sys_data.nbf/10) || tce_tile < 50 || tce_tile > 100) && !sys_data.options_map.ccsd_options.force_tilesize) {
        tce_tile = static_cast<Tile>(sys_data.nbf/10);
        if(tce_tile < 50) tce_tile = 50; //50 is the default tilesize for CCSD.
        if(tce_tile > 100) tce_tile = 100; //100 is the max tilesize for CCSD.
        if(GA_Nodeid()==0) std::cout << std::endl << "Resetting CCSD tilesize to: " << tce_tile << std::endl;
      }
    }
    else tce_tile = sys_data.options_map.ccsd_options.ccsdt_tilesize;

    TAMM_SIZE nmo = sys_data.nmo;
    TAMM_SIZE n_vir_alpha = sys_data.n_vir_alpha;
    TAMM_SIZE n_vir_beta = sys_data.n_vir_beta;
    TAMM_SIZE nocc = sys_data.nocc;
    // TAMM_SIZE nvab = n_vir_alpha+n_vir_beta;

    // std::cout << "n_occ_alpha,nao === " << n_occ_alpha << ":" << nao << std::endl;
    std::vector<TAMM_SIZE> sizes = {n_occ_alpha - freeze_core, n_occ_beta - freeze_core,
             n_vir_alpha - freeze_virtual, n_vir_beta - freeze_virtual};

    const TAMM_SIZE total_orbitals = nmo - 2 * freeze_core - 2 * freeze_virtual;
    
    // cout << "total orb = " <<total_orbitals << endl;
    // cout << "oab = " << n_occ_alpha << endl;
    // cout << "vab = " << ov_beta << endl;

    // Construction of tiled index space MO
    IndexSpace MO_IS{range(0, total_orbitals),
                    {
                     {"occ", {range(0, nocc)}},
                     {"occ_alpha", {range(0, n_occ_alpha)}},
                     {"occ_beta", {range(n_occ_alpha, nocc)}},
                     {"virt", {range(nocc, total_orbitals)}},
                     {"virt_alpha", {range(nocc,nocc+n_vir_alpha)}},
                     {"virt_beta", {range(nocc+n_vir_alpha, total_orbitals)}},                     
                    },
                     { 
                      {Spin{1}, {range(0, n_occ_alpha), range(nocc,nocc+n_vir_alpha)}},
                      {Spin{2}, {range(n_occ_alpha, nocc), range(nocc+n_vir_alpha, total_orbitals)}} 
                     }
                     };

    std::vector<Tile> mo_tiles;
    
    if(!sys_data.options_map.ccsd_options.balance_tiles) {
      tamm::Tile est_nt = n_occ_alpha/tce_tile;
      tamm::Tile last_tile = n_occ_alpha%tce_tile;
      for (tamm::Tile x=0;x<est_nt;x++)mo_tiles.push_back(tce_tile);
      if(last_tile>0) mo_tiles.push_back(last_tile);
      est_nt = n_occ_beta/tce_tile;
      last_tile = n_occ_beta%tce_tile;
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(tce_tile);
      if(last_tile>0) mo_tiles.push_back(last_tile);

      est_nt = n_vir_alpha/tce_tile;
      last_tile = n_vir_alpha%tce_tile;
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(tce_tile);
      if(last_tile>0) mo_tiles.push_back(last_tile);
      est_nt = n_vir_beta/tce_tile;
      last_tile = n_vir_beta%tce_tile;    
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(tce_tile);
      if(last_tile>0) mo_tiles.push_back(last_tile);
    }
    else {
      tamm::Tile est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_occ_alpha / tce_tile));
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(n_occ_alpha / est_nt + (x<(n_occ_alpha % est_nt)));

      est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_occ_beta / tce_tile));
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(n_occ_beta / est_nt + (x<(n_occ_beta % est_nt)));

      est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_vir_alpha / tce_tile));
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(n_vir_alpha / est_nt + (x<(n_vir_alpha % est_nt)));

      est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_vir_beta / tce_tile));
      for (tamm::Tile x=0;x<est_nt;x++) mo_tiles.push_back(n_vir_beta / est_nt + (x<(n_vir_beta % est_nt)));
    }

    // cout << "mo-tiles=" << mo_tiles << endl;

    // IndexSpace MO_IS{range(0, total_orbitals),
    //                 {{"occ", {range(0, n_occ_alpha+ov_beta)}}, //0-7
    //                  {"virt", {range(total_orbitals/2, total_orbitals)}}, //7-14
    //                  {"alpha", {range(0, n_occ_alpha),range(n_occ_alpha+ov_beta,2*n_occ_alpha+ov_beta)}}, //0-5,7-12
    //                  {"beta", {range(n_occ_alpha,n_occ_alpha+ov_beta), range(2*n_occ_alpha+ov_beta,total_orbitals)}} //5-7,12-14   
    //                  }};

    // const unsigned int ova = static_cast<unsigned int>(n_occ_alpha);
    // const unsigned int ovb = static_cast<unsigned int>(ov_beta);
    TiledIndexSpace MO{MO_IS, mo_tiles}; //{ova,ova,ovb,ovb}};

    return std::make_tuple(MO,total_orbitals);
}

template<typename T>
std::tuple<std::vector<T>,Tensor<T>,Tensor<T>,Tensor<T>,Tensor<T>,
std::vector<Tensor<T>>,std::vector<Tensor<T>>,std::vector<Tensor<T>>,std::vector<Tensor<T>>>
 setupTensors(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1, int ndiis, bool ccsd_restart=false) {

    auto rank = ec.pg().rank();

    TiledIndexSpace O = MO("occ");
    TiledIndexSpace V = MO("virt");
 
    std::vector<T> p_evl_sorted = tamm::diagonal(d_f1);

    auto lambda2 = [&](const IndexVector& blockid, span<T> buf){
        if(blockid[0] != blockid[1]) {
            for(auto i = 0U; i < buf.size(); i++) buf[i] = 0; 
        }
    };

    update_tensor(d_f1(),lambda2);

 if(rank == 0) {
    std::cout << std::endl << std::endl;
    std::cout << " CCSD iterations" << std::endl;
    std::cout << std::string(66, '-') << std::endl;
    std::cout <<
        " Iter          Residuum       Correlation     Cpu    Wall    V2*C2"
              << std::endl;
    std::cout << std::string(66, '-') << std::endl;
  }
   
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

    auto lambda2 = [&](const IndexVector& blockid, span<T> buf){
        if(blockid[0] != blockid[1]) {
            for(auto i = 0U; i < buf.size(); i++) buf[i] = 0; 
        }
    };

    update_tensor(d_f1(),lambda2);

 if(rank == 0) {
    std::cout << std::endl << std::endl;
    std::cout << " CCSD iterations" << std::endl;
    std::cout << std::string(66, '-') << std::endl;
    std::cout <<
        " Iter          Residuum       Correlation     Cpu    Wall    V2*C2"
              << std::endl;
    std::cout << std::string(66, '-') << std::endl;
  }
   
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
    double hf_energy{0.0};
    libint2::BasisSet shells;
    Tensor<T> C_AO, C_beta_AO;
    Tensor<T> F_AO, F_beta_AO;
    Tensor<T> ints_3c;
    TiledIndexSpace tAO; //Fixed Tilesize AO
    TiledIndexSpace tAOt; //original AO TIS
    std::vector<size_t> shell_tile_map;
    bool scf_conv;

    // read geometry from a .nwx file 
    auto is = std::ifstream(filename);
    std::vector<Atom> atoms;
    OptionsMap options_map;
    std::tie(atoms, options_map) = read_input_nwx(is);
    if(options_map.options.output_file_prefix.empty()) 
      options_map.options.output_file_prefix = getfilename(filename);
    
    SystemData sys_data{options_map, options_map.scf_options.scf_type};

    auto hf_t1 = std::chrono::high_resolution_clock::now();

    std::tie(sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, C_beta_AO, F_beta_AO, ints_3c, tAO, tAOt, scf_conv) = hartree_fock(ec, filename, atoms, options_map);
    sys_data.input_molecule = getfilename(filename);
    sys_data.output_file_prefix = options_map.options.output_file_prefix;

    auto hf_t2 = std::chrono::high_resolution_clock::now();

    double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    if(rank == 0) std::cout << std::endl << "Time taken for Hartree-Fock: " << hf_time << " secs" << std::endl;

    return std::make_tuple(sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, C_beta_AO, F_beta_AO, tAO, tAOt, scf_conv);
}

#if 0
template<typename T> 
std::tuple<Tensor<T>,Tensor<T>,TAMM_SIZE, tamm::Tile>  cd_svd_driver(OptionsMap options_map,
 ExecutionContext& ec, TiledIndexSpace& MO, TiledIndexSpace& AO_tis,
  const TAMM_SIZE n_occ_alpha, const TAMM_SIZE nao, const TAMM_SIZE freeze_core,
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
    Tensor<T> cholVpr = cd_svd(ec, MO, AO_tis, n_occ_alpha, nao, freeze_core, freeze_virtual,
                                C_AO, F_AO, d_f1, chol_count, max_cvecs, diagtol, shells, shell_tile_map);
    auto hf_t2        = std::chrono::high_resolution_clock::now();
    double cd_svd_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    if(rank == 0) std::cout << std::endl << "Total Time taken for CD (+SVD): " << cd_svd_time
              << " secs" << std::endl;

    Tensor<T>::deallocate(C_AO,F_AO);

    // IndexSpace CI{range(0, max_cvecs)};
    // TiledIndexSpace tCI{CI, max_cvecs};
    // auto [cindex] = tCI.labels<1>("all");

    // IndexSpace CIp{range(0, chol_count)};
    // TiledIndexSpace tCIp{CIp, 1};
    // auto [cindexp] = tCIp.labels<1>("all");

    return std::make_tuple(cholVpr, d_f1, chol_count, max_cvecs);

}
#endif

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
      nwx_terminate("ERROR: CCSD calculation does not converge!");
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
    std::cout << std::string(44, '-') << std::endl;
    std::cout <<
        " Iter          Residuum          Cpu    Wall"
              << std::endl;
    std::cout << std::string(44, '-') << std::endl;
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
Tensor<T> setupV2(ExecutionContext& ec, TiledIndexSpace& MO, TiledIndexSpace& CI,
                  Tensor<T> cholVpr, const tamm::Tile chol_count, 
                  ExecutionHW hw = ExecutionHW::CPU) {

    auto rank = ec.pg().rank();

    TiledIndexSpace N = MO("all");
    auto [cindex] = CI.labels<1>("all");
    auto [p,q,r,s] = MO.labels<4>("all");

    //Spin here is defined as spin(p)=spin(r) and spin(q)=spin(s) which is not currently not supported by TAMM.
    Tensor<T> d_a2{{N,N,N,N},{2,2}};
    //For V2, spin(p)+spin(q) == spin(r)+spin(s)
    Tensor<T> d_v2{{N,N,N,N},{2,2}};
    Tensor<T>::allocate(&ec,d_a2,d_v2);

    auto cc_t1 = std::chrono::high_resolution_clock::now();

    Scheduler{ec}(d_a2(p, q, r, s) = cholVpr(p, r, cindex) * cholVpr(q, s, cindex))
                 (d_v2(p, q, r, s) = d_a2(p,q,r,s))
                 (d_v2(p, q, r, s) -= d_a2(p,q,s,r))
                 .execute(hw);

    auto cc_t2 = std::chrono::high_resolution_clock::now();
    double v2_time = 
        std::chrono::duration_cast<std::chrono::duration<double>>((cc_t2 - cc_t1)).count();
    if(rank == 0) std::cout << std::endl << "Time to reconstruct V2: " << v2_time << " secs" << std::endl;

    Tensor<T>::deallocate(d_a2);
    return d_v2;

 #if 0
      auto chol_dims = CholVpr.dimensions();
  auto chol_count = chol_dims[2];
    auto ndocc = n_occ_alpha;
    auto n_occ_alpha_freeze = ndocc - freeze_core;
    auto ov_beta_freeze  = nao - ndocc - freeze_virtual;
  const int n_alpha = n_occ_alpha_freeze;
  const int n_beta = ov_beta_freeze;
  // buf[0] points to the target shell set after every call  to engine.compute()
  // const auto &buf = engine.results();
  Matrix spin_t = Matrix::Zero(1, 2 * nao - 2 * freeze_core - 2 * freeze_virtual);
  Matrix spin_1 = Matrix::Ones(1,n_alpha);
  Matrix spin_2 = Matrix::Constant(1,n_alpha,2);
  Matrix spin_3 = Matrix::Constant(1,n_beta,1);
  Matrix spin_4 = Matrix::Constant(1,n_beta,2);
  //spin_t << 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2; - water
  spin_t.block(0,0,1,n_alpha) = spin_1;
  spin_t.block(0,n_alpha,1,n_alpha) = spin_2;
  spin_t.block(0,2*n_alpha,1, n_beta) = spin_3;
  spin_t.block(0,2*n_alpha+n_beta,1, n_beta) = spin_4;

  const auto v2dim =  2 * nao - 2 * freeze_core - 2 * freeze_virtual;
    Tensor4D A2(v2dim,v2dim,v2dim,v2dim);
    A2.setZero();
    Tensor4D V2(v2dim,v2dim,v2dim,v2dim);

        // Form (pr|qs)
    for (auto p = 0; p < v2dim; p++) {
      for (auto r = 0; r < v2dim; r++) {
        if (spin_t(p) != spin_t(r)) {
          continue;
        }

        for (auto q = 0; q < v2dim; q++) {
          for (auto s = 0; s < v2dim; s++) {
            if (spin_t(q) != spin_t(s)) {
              continue;
            }

            for (auto icount = 0; icount != chol_count; ++icount) {
              A2(p, r, q, s) += CholVpr(p, r, icount) * CholVpr(q, s, icount);
              //V2_FromCholV(p, r, q, s) += CholVpr(p, r, icount) * CholVpr(q, s, icount);
            }
            //cout << p << " " << r << " " << q << " " << s << " " << V2_unfused(p, r, q, s) << endl << endl;
          }
        }
      }
    }


    for (size_t p = 0; p < v2dim; p++) {
        for (size_t q = 0; q < v2dim; q++) {
          for (size_t r = 0; r < v2dim; r++) {
            for (size_t s = 0; s < v2dim; s++) {
              V2(p, q, r, s) = A2(p, r, q, s) - A2(p, s, q, r);
            }
          }
        }
    }
 #endif

}

template<typename T> 
std::tuple<Tensor<T>,Tensor<T>,TAMM_SIZE, tamm::Tile, TiledIndexSpace>  cd_svd_ga_driver(SystemData& sys_data,
  ExecutionContext& ec, TiledIndexSpace& MO, TiledIndexSpace& AO_tis,
  Tensor<TensorType> C_AO, Tensor<TensorType> F_AO, Tensor<TensorType> C_beta_AO, Tensor<TensorType> F_beta_AO, libint2::BasisSet& shells, 
  std::vector<size_t>& shell_tile_map, bool readv2=false, std::string cholfile=""){

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
    Tensor<T>::allocate(&ec,d_f1);

    auto hf_t1        = std::chrono::high_resolution_clock::now();
    TAMM_SIZE chol_count = 0;

    //std::tie(V2) = 
    Tensor<T> cholVpr;

    auto itile_size = sys_data.options_map.ccsd_options.itilesize;

    if(!readv2) {
      cholVpr = cd_svd_ga(sys_data, ec, MO, AO_tis,
                          C_AO, F_AO, C_beta_AO,F_beta_AO, d_f1, chol_count, max_cvecs, shells, shell_tile_map);
    }
    else{
      std::ifstream in(cholfile, std::ios::in);
      int rstatus = 0;
      if(in.is_open()) rstatus = 1;
      if(rstatus == 1) in >> chol_count; 
      else nwx_terminate("Error reading " + cholfile);

      if(rank==0) cout << "Number of cholesky vectors to be read = " << chol_count << endl;

      IndexSpace chol_is{range(0,chol_count)};
      TiledIndexSpace CI{chol_is,static_cast<tamm::Tile>(itile_size)}; 

      cholVpr = {{N,N,CI},{SpinPosition::upper,SpinPosition::lower,SpinPosition::ignore}};
      Tensor<TensorType>::allocate(&ec, cholVpr);
      // Scheduler{ec}(cholVpr()=0).execute();
    }

    auto hf_t2        = std::chrono::high_resolution_clock::now();
    double cd_svd_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    if(rank == 0) std::cout << std::endl << "Total Time taken for CD (+SVD): " << cd_svd_time
              << " secs" << std::endl;

    Tensor<T>::deallocate(C_AO,F_AO);
    if(sys_data.scf_type == sys_data.SCFType::uhf) Tensor<T>::deallocate(C_beta_AO,F_beta_AO);

    IndexSpace chol_is{range(0,chol_count)};
    TiledIndexSpace CI{chol_is,static_cast<tamm::Tile>(itile_size)}; 

    sys_data.num_chol_vectors = chol_count;
    if(rank == 0) sys_data.print();

    return std::make_tuple(cholVpr, d_f1, chol_count, max_cvecs, CI);
}

#endif //TESTS_CCSD_UTIL_HPP_
