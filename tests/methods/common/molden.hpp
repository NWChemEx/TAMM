
#ifndef METHODS_MOLDEN_HPP_
#define METHODS_MOLDEN_HPP_

#include <iostream>
#include "input_parser.hpp"


template<typename T>
std::tuple<int,int,int,int> read_mo(SCFOptions scf_options, std::istream& is, std::vector<T>& evl_sorted, 
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& C) {

  int n_occ_alpha=0, n_occ_beta=0, n_vir_alpha=0, n_vir_beta=0;
    
  std::string line;
  bool mo_end = false;
  size_t nmo = evl_sorted.size();
  size_t N = nmo;
  if(scf_options.scf_type == "uhf") N = N/2;
  size_t i = 0;
  size_t kb = 0;
  const size_t n_lindep = scf_options.n_lindep;

  while(!mo_end){
    
    std::getline(is, line);
    if (line.find("Ene=") != std::string::npos) {
      evl_sorted[i] = (std::stod(read_option(line)));
    }
    else if (line.find("Spin=") != std::string::npos){
      std::string spinstr = read_option(line);
      bool is_spin_alpha = spinstr.find("Alpha") != std::string::npos;
      bool is_spin_beta = spinstr.find("Beta") != std::string::npos;
      
      // if(is_spin_alpha) n_alpha++;
      // else if(is_spin_beta) n_beta++;
      std::getline(is, line);

      if (line.find("Occup=") != std::string::npos){
        int occup = stoi(read_option(line));
        if(is_spin_alpha) {
          if(occup==0) n_vir_alpha++;
          if(occup==1) n_occ_alpha++;
          if(occup==2) { n_occ_alpha++; n_occ_beta++; }
        }
        else if(is_spin_beta){
           if(occup==1) n_occ_beta++;
           if(occup==0) n_vir_beta++; 
        }
        mo_end=true;
      }
    }

    if(mo_end){
      for(size_t j=0;j<N;j++){
        std::getline(is, line);
        C(j,i) = std::stod(read_option(line));
      }
      mo_end=false;
      i++;
      if(i==N-n_lindep) i=i+n_lindep;
    }

    kb++;
    if(i==nmo-n_lindep) mo_end=true;
    if(kb==4*nmo) {
      // cout << "Assuming n_lindep = " << nmo-i << endl;
      mo_end=true;
    }
    
  }

  if(scf_options.scf_type == "rhf") { 
      n_occ_beta = n_occ_alpha;
      n_vir_beta = n_vir_alpha;
  }
  else if(scf_options.scf_type == "rohf") { 
      n_vir_beta = N - n_occ_beta;
  }

  // cout << "finished rading modlden: n_occ_alpha, n_vir_alpha, n_occ_beta, n_vir_beta = " 
  //        << n_occ_alpha << "," << n_vir_alpha << "," << n_occ_beta << "," << n_vir_beta << endl;

  return std::make_tuple(n_occ_alpha,n_vir_alpha,n_occ_beta,n_vir_beta);
   
}

template<typename T>
std::tuple<int,int,int,int> read_molden(SCFOptions scf_options, std::vector<T>& evl_sorted, 
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& C) {

  auto is = std::ifstream(scf_options.moldenfile);
  std::string line;
  bool mo_start = false;
  int n_occ_alpha=0, n_occ_beta=0, n_vir_alpha=0, n_vir_beta=0;
  
  while(!mo_start) {
    //skip_empty_lines(is);
    std::getline(is, line);

    if(is_in_line("[MO]",line)){
      mo_start = true;
      std::tie(n_occ_alpha,n_occ_beta,n_vir_alpha,n_vir_beta) = read_mo(scf_options, is,evl_sorted,C);
    }
  }

  // std::cout << "#alpha:" << n_alpha << std::endl;
  // std::cout << "evl-sorted:" << std::endl;
  // for (auto x: evl_sorted)
  // std::cout << x << std::endl;
  // std::cout << "movecs\n";
  // std::cout << C << std::endl;

  return std::make_tuple(n_occ_alpha,n_occ_beta,n_vir_alpha,n_vir_beta);

}

#endif // METHODS_MOLDEN_HPP_
