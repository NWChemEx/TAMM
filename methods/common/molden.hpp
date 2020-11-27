
#ifndef METHODS_MOLDEN_HPP_
#define METHODS_MOLDEN_HPP_

#include <iostream>
#include "input_parser.hpp"

string read_option(string line){
  std::istringstream oss(line);
  std::vector<std::string> option_string{
    std::istream_iterator<std::string>{oss},
    std::istream_iterator<std::string>{}};
  // assert(option_string.size() == 2);
  
  return option_string[1];
}

bool is_comment(const std::string line) {
  auto found = false;
  if(line.find("//") != std::string::npos){
    // found = true;
    auto fpos = line.find_first_not_of(' ');
    auto str = line.substr(fpos,2);
    if (str == "//") found = true;
  }
  return found;
}

bool is_in_line(const std::string str, const std::string line){
  auto found = true;
  std::string str_u = str, str_l = str;
  to_upper(str_u); to_lower(str_l);

  if (is_comment(line)) found = false;
  else {
    std::istringstream oss(line);
    std::vector<std::string> option_string{
    std::istream_iterator<std::string>{oss},
    std::istream_iterator<std::string>{}};
    for (auto &x: option_string) 
      x.erase(std::remove(x.begin(),x.end(),' '),x.end());
    
    if (std::find(option_string.begin(),option_string.end(), str_u) == option_string.end()
     && std::find(option_string.begin(),option_string.end(), str_l) == option_string.end() )
     found = false;
  }

  return found;
}

// bool is_empty(std::string line){
//   if(line.find_first_not_of(' ') == std::string::npos 
//     || line.empty() || is_comment(line)) return true;
//   return false;
// }

// void skip_empty_lines(std::istream& is) {
//     std::string line;
//     auto curpos = is.tellg();
//     std::getline(is, line);
    
//     if(is_empty(line)) {
//       curpos = is.tellg();
//       auto trackpos = curpos;
//       while (is_empty(line)) {
//         curpos = trackpos;
//         std::getline(is, line);
//         trackpos = is.tellg();
//       }
//     }
//     is.clear();//cannot seek to curpos if eof is reached
//     is.seekg(curpos,std::ios_base::beg);
//     // std::getline(is, line);
// }

template<typename T>
std::tuple<int,int,int,int> read_mo(SCFOptions scf_options, std::istream& is, std::vector<T>& evl_sorted, 
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& C, const size_t natoms) {

  int n_occ_alpha=0, n_occ_beta=0, n_vir_alpha=0, n_vir_beta=0;
    
  std::string line;
  
  size_t nmo = evl_sorted.size();
  size_t N = C.rows(); //nmo
  //if(scf_options.scf_type == "uhf") N = N/2;
  //const size_t n_lindep = scf_options.n_lindep;
  std::vector<T> eigenvecs(N); 
  bool is_spherical = (scf_options.sphcart == "spherical");

  size_t s_count = 0;
  size_t p_count = 0;
  size_t d_count = 0;
  size_t f_count = 0;
  size_t g_count = 0;

  //s_type = 0, p_type = 1, d_type = 2, 
  //f_type = 3, g_type = 4
  /*For spherical
  n = 2*type + 1
  s=1,p=3,d=5,f=7,g=9 
  For cartesian
  n = (type+1)*(type+2)/2
  s=1,p=3,d=6,f=10,g=15
  */

  bool basis_end = false;
  while(!basis_end) {
    std::getline(is, line);
    if (line.find("GTO") != std::string::npos) {
      basis_end=true;
      bool basis_parse=true;
      while(basis_parse) {
        std::getline(is, line);
        std::istringstream iss(line);
        std::vector<std::string> atom2{std::istream_iterator<std::string>{iss},
                                       std::istream_iterator<std::string>{}};
        if(atom2.size()==0) continue;
        if(atom2.size()==2)
          {if(atom2[0]=="2" && atom2[1]=="0") basis_parse=false;}
                                  
        if (line.find("[5D]")   != std::string::npos) basis_parse=false;
        if (atom2[0]=="s") s_count++;
        else if (atom2[0]=="p") p_count+=3;
        else if (atom2[0]=="d") {
          if(is_spherical) d_count+=5;
          else d_count+=6;
        }
        else if (atom2[0]=="f") {
          if(is_spherical) f_count+=7;
          else f_count+=10;
        }
        else if (atom2[0]=="g") {
          if(is_spherical) g_count+=9;
          else g_count+=15;
        }
      } //end 
      
    } //end basis parse
  } //end basis section

  // cout << "finished reading basis: s_count, p_count, d_count, f_count, g_count = " 
  //      << s_count << "," << p_count << "," << d_count << "," << f_count << "," << g_count <<  endl;

  const size_t sp_count = s_count+p_count;
  const size_t spd_count = sp_count+d_count;
  const size_t spdf_count = spd_count+f_count;
  const size_t spdfg_count = spdf_count+g_count;
  //if(spdfg_count*natoms != N) tamm_terminate("Moldenfile read error");
  const T sqrt_3 = std::sqrt(static_cast<T>(3.0));
  const T sqrt_5 = std::sqrt(static_cast<T>(5.0));
  const T sqrt_7 = std::sqrt(static_cast<T>(7.0));
  const T sqrt_753 = sqrt_7*sqrt_5/sqrt_3;

  bool mo_start = false;
  while(!mo_start) {
    //skip_empty_lines(is);
    std::getline(is, line);

    if(is_in_line("[MO]",line))
      mo_start = true;
  }
  
  bool mo_end = false;
  size_t i = 0;
  // size_t kb = 0;
  while(!mo_end) { 
    
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
        else if(is_spin_beta) {
           if(occup==1) n_occ_beta++;
           if(occup==0) n_vir_beta++; 
        }
        mo_end=true;
      }
    }

    if(mo_end){
      for(size_t j=0;j<N;j++) {
        std::getline(is, line);
        eigenvecs[j] = std::stod(read_option(line));
      }
      //reorder
      if(is_spherical) {
        for(size_t j=0;j<N;j++) {
          const size_t j_a = j%spdfg_count; //for each atom

          if(j_a < s_count) //S functions
            C(j,i) = eigenvecs[j];
          else if(j_a >= s_count && j_a < sp_count) {
            //P functions
            //libint set_pure to solid forces y,z,x ordering for l=1
            C(j,i) = eigenvecs[j+1]; j++;
            C(j,i) = eigenvecs[j+1]; j++;
            C(j,i) = eigenvecs[j-2]; 
          }
          else if(j_a >= sp_count && j_a < spd_count) {
            //D functions
            C(j,i) = eigenvecs[j+4]; j++;
            C(j,i) = eigenvecs[j+1]; j++;
            C(j,i) = eigenvecs[j-2]; j++;
            C(j,i) = eigenvecs[j-2]; j++;
            C(j,i) = eigenvecs[j-1]; 
          }
          else if(j_a >= spd_count && j_a < spdf_count) {
            //F functions
            C(j,i) = eigenvecs[j+6]; j++;
            C(j,i) = eigenvecs[j+3]; j++;
            C(j,i) = eigenvecs[j  ]; j++;
            C(j,i) = eigenvecs[j-3]; j++;
            C(j,i) = eigenvecs[j-3]; j++;
            C(j,i) = eigenvecs[j-2]; j++;
            C(j,i) = eigenvecs[j-1];
          }
          else if(j_a >= spdf_count && j_a < spdfg_count) {
            //G functions
            C(j,i) = eigenvecs[j+8]; j++;
            C(j,i) = eigenvecs[j+5]; j++;
            C(j,i) = eigenvecs[j+2]; j++;
            C(j,i) = eigenvecs[j-1]; j++;
            C(j,i) = eigenvecs[j-4]; j++;
            C(j,i) = eigenvecs[j-4]; j++;
            C(j,i) = eigenvecs[j-3]; j++;
            C(j,i) = eigenvecs[j-2]; j++;
            C(j,i) = eigenvecs[j-1];           
          }
        }
      }
      else {
        //TODO cartesian f,g
        for(size_t j=0;j<N;j++) {
          const size_t j_a = j%spdfg_count; //for each atom

          if(j_a < s_count) //S functions
            C(j,i) = eigenvecs[j];
          else if(j_a >= s_count && j_a < sp_count) //P functions
            C(j,i) = eigenvecs[j];
          else if(j_a >= sp_count && j_a < spd_count) {
            //D functions
            C(j,i) = eigenvecs[j];          j++;
            C(j,i) = eigenvecs[j+2]*sqrt_3; j++;
            C(j,i) = eigenvecs[j+2]*sqrt_3; j++;
            C(j,i) = eigenvecs[j-2];        j++;
            C(j,i) = eigenvecs[j+1]*sqrt_3; j++;
            C(j,i) = eigenvecs[j-3];        
          }
          else if(j_a >= spd_count && j_a < spdf_count) {
            //F functions
            C(j,i) = eigenvecs[j];                 j++;
            C(j,i) = eigenvecs[j+3]*sqrt_5;        j++;
            C(j,i) = eigenvecs[j+3]*sqrt_5;        j++;
            C(j,i) = eigenvecs[j  ]*sqrt_5;        j++;
            C(j,i) = eigenvecs[j+5]*sqrt_5*sqrt_3; j++;
            C(j,i) = eigenvecs[j+1]*sqrt_5;        j++;
            C(j,i) = eigenvecs[j-5];               j++;
            C(j,i) = eigenvecs[j+1]*sqrt_5;        j++;
            C(j,i) = eigenvecs[j-1]*sqrt_5;        j++;
            C(j,i) = eigenvecs[j-7];
          }
          else if(j_a >= spdf_count && j_a < spdfg_count) {
            //G functions
            C(j,i) = eigenvecs[j];                 j++;
            C(j,i) = eigenvecs[j+2]*sqrt_7;        j++;
            C(j,i) = eigenvecs[j+2]*sqrt_7;        j++;
            C(j,i) = eigenvecs[j+6]*sqrt_753;      j++;
            C(j,i) = eigenvecs[j+8]*sqrt_7*sqrt_5; j++;
            C(j,i) = eigenvecs[j+5]*sqrt_753;      j++;
            C(j,i) = eigenvecs[j-1]*sqrt_7;        j++;
            C(j,i) = eigenvecs[j+6]*sqrt_7*sqrt_5; j++;
            C(j,i) = eigenvecs[j+6]*sqrt_7*sqrt_5; j++;
            C(j,i) = eigenvecs[j-2]*sqrt_7;        j++;
            C(j,i) = eigenvecs[j-9];               j++;
            C(j,i) = eigenvecs[j-5]*sqrt_7;        j++;
            C(j,i) = eigenvecs[j-1]*sqrt_753;      j++;
            C(j,i) = eigenvecs[j-5]*sqrt_7  ;      j++;
            C(j,i) = eigenvecs[j-12];                  
          }
        }        
      }

      mo_end=false;
      i++;
      //if(i==N-n_lindep) i=i+n_lindep;
    }

    // kb++;
    // if(i==nmo-n_lindep) mo_end=true;
    if(i==nmo) mo_end=true;
    // if(kb==4*nmo) {
    //   // cout << "Assuming n_lindep = " << nmo-i << endl;
    //   mo_end=true;
    // }
    
  }

  if(scf_options.scf_type == "rhf") { 
      n_occ_beta = n_occ_alpha;
      n_vir_beta = n_vir_alpha;
  }
  else if(scf_options.scf_type == "rohf") { 
      n_vir_beta = N - n_occ_beta;
  }

  // cout << "finished reading molden: n_occ_alpha, n_vir_alpha, n_occ_beta, n_vir_beta = " 
  //        << n_occ_alpha << "," << n_vir_alpha << "," << n_occ_beta << "," << n_vir_beta << endl;

  return std::make_tuple(n_occ_alpha,n_vir_alpha,n_occ_beta,n_vir_beta);
   
}

template<typename T>
std::tuple<int,int,int,int> read_molden(SCFOptions scf_options, std::vector<T>& evl_sorted, 
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& C, const size_t natoms) {

  auto is = std::ifstream(scf_options.moldenfile);
  std::string line;
  int n_occ_alpha=0, n_occ_beta=0, n_vir_alpha=0, n_vir_beta=0;
  
  std::tie(n_occ_alpha,n_occ_beta,n_vir_alpha,n_vir_beta) = read_mo(scf_options,is,evl_sorted,C,natoms);

  // std::cout << "#alpha:" << n_alpha << std::endl;
  // std::cout << "evl-sorted:" << std::endl;
  // for (auto x: evl_sorted)
  // std::cout << x << std::endl;
  // std::cout << "movecs" << std::endl;
  // std::cout << C << std::endl;

  return std::make_tuple(n_occ_alpha,n_occ_beta,n_vir_alpha,n_vir_beta);

}

#endif // METHODS_MOLDEN_HPP_
