
#ifndef TAMM_TWO_INDEX_TRANSFORM_HPP_
#define TAMM_TWO_INDEX_TRANSFORM_HPP_

#include "tamm/eigen_utils.hpp"
using namespace tamm;

template <typename TensorType>
void two_index_transform(SystemData sys_data, ExecutionContext& ec, Tensor<TensorType> C_alpha_AO, 
  Tensor<TensorType> F_alpha_AO, Tensor<TensorType> C_beta_AO, Tensor<TensorType> F_beta_AO, 
  Tensor<TensorType> F_MO, Tensor<TensorType> lcao, bool isdlpno=false) {

  SCFOptions         scf_options    = sys_data.options_map.scf_options;
  const TAMM_GA_SIZE n_occ_alpha    = sys_data.n_occ_alpha;
  const TAMM_GA_SIZE n_occ_beta     = sys_data.n_occ_beta;
  const TAMM_GA_SIZE n_vir_alpha    = sys_data.n_vir_alpha;
  const TAMM_GA_SIZE n_vir_beta     = sys_data.n_vir_beta;
  const TAMM_GA_SIZE northo         = sys_data.nbf;
  const TAMM_GA_SIZE nao            = sys_data.nbf_orig;

  auto rank = ec.pg().rank();

  //
  // 2-index transform
  // 
  auto hf_t1 = std::chrono::high_resolution_clock::now();

  auto MO              = F_MO.tiled_index_spaces()[0];
  TAMM_GA_SIZE N       = MO("all").max_num_indices();
  TAMM_GA_SIZE nmo_occ = MO("occ").max_num_indices();
  
  const bool is_uhf = sys_data.is_unrestricted;
  const bool is_rhf = sys_data.is_restricted;
  // const bool is_rohf = sys_data.is_restricted_os;

  std::string out_fp = sys_data.output_file_prefix+"."+sys_data.options_map.ccsd_options.basis;
  std::string files_dir = out_fp+"_files/"+sys_data.options_map.scf_options.scf_type;
  std::string files_prefix = /*out_fp;*/ files_dir+"/"+out_fp;
  std::string lcaofile = files_prefix+".lcao";

  Matrix CTiled(nao, N);

  if(rank == 0) {
    cout << std::endl << "-----------------------------------------------------" << endl;
    cout << "Begin 2-index transformation ... " << endl;
    cout << std::endl << "nAO, nMO, nelectrons = " << nao << ", " << N << ", " << n_occ_alpha+n_occ_beta << endl;

    if(!isdlpno) {
      Matrix C_alpha_eig;
      Matrix C_beta_eig;
      C_alpha_eig.setZero(nao,northo);
      tamm_to_eigen_tensor(C_alpha_AO, C_alpha_eig);
      if(is_uhf) {
        C_beta_eig.setZero(nao,northo);
        tamm_to_eigen_tensor(C_beta_AO, C_beta_eig);
      }
      // // replicate horizontally
      Matrix C_2N(nao, N);
      if(is_rhf) C_2N << C_alpha_eig, C_alpha_eig;
      else if(is_uhf) C_2N << C_alpha_eig, C_beta_eig;
      

      cout << "n_occ_alpha, n_vir_alpha, n_occ_beta, n_vir_beta = " 
          << n_occ_alpha << "," << n_vir_alpha << "," << n_occ_beta << "," << n_vir_beta << endl;

      Matrix C_noa = C_2N.block(0, 0,                   nao, n_occ_alpha);
      Matrix C_nva = C_2N.block(0, n_occ_alpha,         nao, n_vir_alpha);
      Matrix C_nob = C_2N.block(0, northo,              nao, n_occ_beta);
      Matrix C_nvb = C_2N.block(0, northo + n_occ_beta, nao, n_vir_beta);

      C_2N.resize(0,0);
      CTiled << C_noa, C_nob, C_nva, C_nvb;

      Matrix F_alpha_AO_eig = tamm_to_eigen_matrix(F_alpha_AO);
      Matrix F_beta_AO_eig;
      if(is_uhf) F_beta_AO_eig = tamm_to_eigen_matrix(F_beta_AO);

      Matrix F_MO_alpha = C_alpha_eig.transpose() * (F_alpha_AO_eig * C_alpha_eig);
      Matrix F_MO_beta = F_MO_alpha;
      if(is_uhf) F_MO_beta = C_beta_eig.transpose() * (F_beta_AO_eig * C_beta_eig);

      Matrix F;
      F.setZero(N,N);
      /*
        F_MO_alpha = | F_MO_alpha_oo, F_MO_alpha_ov |   F_MO_beta = | F_MO_beta_oo, F_MO_beta_ov | 
                     | F_MO_alpha_vo, F_MO_alpha_vv |               | F_MO_beta_vo, F_MO_beta_vv |
        
        F_MO       = |  F_MO_alpha_oo           0             F_MO_alpha_ov            0         |
                     |      0               F_MO_beta_oo            0             F_MO_beta_ov   |
                     |  F_MO_alpha_vo           0             F_MO_alpha_vv            0         |
                     |      0               F_MO_beta_vo            0             F_MO_beta_vv   |
      */
      F.block(0,       0,        n_occ_alpha, n_occ_alpha) = F_MO_alpha.block(0,                     0, n_occ_alpha, n_occ_alpha);
      F.block(0,       nmo_occ,  n_occ_alpha, n_vir_alpha) = F_MO_alpha.block(0,           n_occ_alpha, n_occ_alpha, n_vir_alpha);
      F.block(nmo_occ, 0,        n_vir_alpha, n_occ_alpha) = F_MO_alpha.block(n_occ_alpha,           0, n_vir_alpha, n_occ_alpha);
      F.block(nmo_occ, nmo_occ,  n_vir_alpha, n_vir_alpha) = F_MO_alpha.block(n_occ_alpha, n_occ_alpha, n_vir_alpha, n_vir_alpha);

      F.block(n_occ_alpha,         n_occ_alpha,         n_occ_beta, n_occ_beta) = F_MO_beta.block(0,                   0, n_occ_beta, n_occ_beta);
      F.block(n_occ_alpha,         nmo_occ+n_vir_alpha, n_occ_beta, n_vir_beta) = F_MO_beta.block(0,          n_occ_beta, n_occ_beta, n_vir_beta);
      F.block(nmo_occ+n_vir_alpha, n_occ_alpha,         n_vir_beta, n_occ_beta) = F_MO_beta.block(n_occ_beta,          0, n_vir_beta, n_occ_beta);
      F.block(nmo_occ+n_vir_alpha, nmo_occ+n_vir_alpha, n_vir_beta, n_vir_beta) = F_MO_beta.block(n_occ_beta, n_occ_beta, n_vir_beta, n_vir_beta);

      eigen_to_tamm_tensor(F_MO,F);
      eigen_to_tamm_tensor(lcao,CTiled);
      write_scf_mat<TensorType>(CTiled,lcaofile);
    }
    else {
      Matrix F_AO_eig,F_MO_eig,C_AO_eig;
      F_AO_eig.setZero(nao,nao);
      C_AO_eig.setZero(nao,N);

      tamm_to_eigen_tensor(F_alpha_AO,F_AO_eig);
      tamm_to_eigen_tensor(C_alpha_AO,C_AO_eig);
      F_MO_eig = C_AO_eig.transpose() * (F_AO_eig * C_AO_eig);
      eigen_to_tamm_tensor(F_MO,F_MO_eig);
      eigen_to_tamm_tensor(lcao,C_AO_eig);

    }

  }

  GA_Sync();  

  auto hf_t2 = std::chrono::high_resolution_clock::now();
  auto hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
  if(rank == 0) {
    std::cout << std::endl << "Time taken for Fao->Fmo transform: " << hf_time << " secs" << endl;
    cout << std::endl << "-----------------------------------------------------" << endl;
  }
}

#endif //TAMM_TWO_INDEX_TRANSFORM_HPP_
