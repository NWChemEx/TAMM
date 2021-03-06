
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
  const TAMM_GA_SIZE freeze_core    = sys_data.n_frozen_core;
  const TAMM_GA_SIZE freeze_virtual = sys_data.n_frozen_virtual;
  // const TAMM_GA_SIZE n_lindep       = sys_data.n_lindep;

  auto rank = ec.pg().rank();

  auto n_occ_alpha_eff = n_occ_alpha - freeze_core;
  auto n_vir_alpha_eff = n_vir_alpha - freeze_virtual;
  auto n_occ_beta_eff  = n_occ_beta  - freeze_core;
  auto n_vir_beta_eff  = n_vir_beta  - freeze_virtual;
  auto n_occ_eff       = n_occ_alpha_eff + n_occ_beta_eff;

  //
  // 2-index transform
  // 
  auto hf_t1 = std::chrono::high_resolution_clock::now();

  auto MO = F_MO.tiled_index_spaces()[0];
  TAMM_GA_SIZE N = MO("all").max_num_indices();
  
  const bool is_rhf  = scf_options.scf_type == "rhf";
  const bool is_uhf  = scf_options.scf_type == "uhf";
  // const bool is_rohf = scf_options.scf_type == "rohf";

  Matrix CTiled(nao, N);

  if(rank == 0) {
    cout << std::endl << "-----------------------------------------------------" << endl;
    cout << "Begin 2-index transformation ... " << endl;
    cout << std::endl << "nAO, nMO, nelectrons = " << nao << ", " << N << ", " << n_occ_alpha+n_occ_beta << endl;

    if(!isdlpno) {
      Matrix C_alpha;
      Matrix C_beta;
      C_alpha.setZero(nao,northo);
      tamm_to_eigen_tensor(C_alpha_AO,C_alpha);
      // replicate horizontally
      Matrix C_2N(nao, N);
      if(is_rhf) C_2N << C_alpha, C_alpha;
      if(is_uhf) {
        C_beta.setZero(nao,northo);
        tamm_to_eigen_tensor(C_beta_AO, C_beta);
        C_2N << C_alpha, C_beta;
        C_beta.resize(0,0);
      }
      C_alpha.resize(0,0);

      cout << "n_occ_alpha, n_vir_alpha, n_occ_beta, n_vir_beta = " 
          << n_occ_alpha << "," << n_vir_alpha << "," << n_occ_beta << "," << n_vir_beta << endl;

      Matrix C_noa = C_2N.block(0, 0,                       nao, n_occ_alpha_eff);
      Matrix C_nva = C_2N.block(0, n_occ_alpha_eff,         nao, n_vir_alpha_eff);
      Matrix C_nob = C_2N.block(0, northo,                  nao, n_occ_beta_eff);
      Matrix C_nvb = C_2N.block(0, northo + n_occ_beta_eff, nao, n_vir_beta_eff);

      C_2N.resize(0,0);
      CTiled << C_noa, C_nob, C_nva, C_nvb;

      Matrix F1_a;
      Matrix F1_b;
      F1_a.setZero(nao,nao);
      tamm_to_eigen_tensor(F_alpha_AO,F1_a);

      if(is_uhf) {
        F1_b.setZero(nao,nao);
        tamm_to_eigen_tensor(F_beta_AO, F1_b);
      }

      Matrix F_oa;
      Matrix F_va;
      Matrix F_ob;
      Matrix F_vb;
      // F_oa.setZero(n_occ_alpha_eff,n_occ_alpha_eff);
      // F_va.setZero(n_vir_alpha_eff,n_vir_alpha_eff);
      // F_ob.setZero(n_occ_beta_eff, n_occ_beta_eff);
      // F_vb.setZero(n_vir_beta_eff, n_vir_beta_eff);

      F_oa = C_noa.transpose() * (F1_a * C_noa);
      F_va = C_nva.transpose() * (F1_a * C_nva);

      if(is_rhf) {
        F_ob = F_oa;
        F_vb = F_va;
      }
      if(is_uhf) {
        F_ob = C_nob.transpose() * (F1_b * C_nob);
        F_vb = C_nvb.transpose() * (F1_b * C_nvb);
      }

      F1_a.resize(0,0);
      F1_b.resize(0,0);

      Matrix F;
      F.setZero(N,N);

      TAMM_GA_SIZE k = 0;
      for (TAMM_GA_SIZE i=0;i<n_occ_alpha_eff;i++){
        F(i,i) = F_oa(k,k);
        k++;
      }
      k = 0;
      for (TAMM_GA_SIZE i=n_occ_alpha_eff;i<n_occ_eff;i++){
        F(i,i) = F_ob(k,k);
        k++;
      }
      k = 0;
      for (TAMM_GA_SIZE i=n_occ_eff;i<n_occ_eff+n_vir_alpha_eff;i++){
        F(i,i) = F_va(k,k);
        k++;
      }
      k = 0;
      for (TAMM_GA_SIZE i=n_occ_eff+n_vir_alpha_eff;i<N;i++){
        F(i,i) = F_vb(k,k);
        k++;
      }

      eigen_to_tamm_tensor(F_MO,F);
      eigen_to_tamm_tensor(lcao,CTiled);
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
