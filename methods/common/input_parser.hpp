
#ifndef TAMM_METHODS_INPUT_PARSER_HPP_
#define TAMM_METHODS_INPUT_PARSER_HPP_

#include <cctype>
#include <string>
#include <vector>
#include <iostream>
#include <unordered_map>


// Libint Gaussian integrals library
#include <libint2.hpp>
#include <libint2/basis.h>
#include <libint2/chemistry/sto3g_atomic_density.h>

#include "ga.h"
#include "ga-mpi.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using libint2::Atom;

inline bool strequal_case( const std::string &a, const std::string &b ) {
  return a.size() == b.size() and
    std::equal( a.begin(), a.end(), b.begin(), [](const char a, const char b) {
      return std::tolower(a) == std::tolower(b);
    });
}

// const int nwx_max_section_options = 20;

void print_bool(std::string str, bool val){
  if(val) cout << str << " = true" << endl;
  else cout << str << " = false" << endl;    
}

class Options {

   public:
    Options(){
      maxiter = 50;
      debug = false;
      basis = "sto-3g";
      dfbasis = "";
      geom_units = "bohr";
      sphcart = "spherical";
      output_file_prefix = "";
    }

    bool debug;
    int maxiter;
    std::string basis;
    std::string dfbasis;
    std::string sphcart;
    std::string geom_units;
    std::string output_file_prefix;

    void print() {
      std::cout << std::defaultfloat;
      cout << endl << "Common Options" << endl ;
      cout << "{" << endl;
      cout << " maxiter    = " << maxiter << endl;
      cout << " basis      = " << basis << " ";
      cout << sphcart;
      cout << endl;
      if(!dfbasis.empty()) 
        cout << " dfbasis    = " << dfbasis << endl;       
      cout << " geom_units = " << geom_units << endl;
      print_bool(" debug     ", debug);
      if(!output_file_prefix.empty()) 
        cout << " output_file_prefix    = " << output_file_prefix << endl;       
      cout << "}" << endl;
    }
};

class SCFOptions: public Options {

  public:
    SCFOptions() = default;
    SCFOptions(Options o): Options(o)
    {
      charge         = 0;
      multiplicity   = 1;
      lshift         = 0;
      tol_int        = 1e-12;
      tol_lindep     = 1e-5;
      conve          = 1e-8;
      convd          = 1e-6;
      diis_hist      = 10;
      AO_tilesize    = 30;
      restart        = false;
      noscf          = false;
      ediis          = false;
      ediis_off      = 1e-5;
      sad            = false;
      force_tilesize = false;
      riscf          = 0; //0 for JK, 1 for J, 2 for K
      riscf_str      = "JK";
      moldenfile     = "";
      n_lindep       = 0;
      scf_type       = "rhf";
      alpha          = 0.7;
      nnodes         = 1;
      writem         = diis_hist;
      scalapack_nb   = 1;
      scalapack_np_row   = 0;
      scalapack_np_col   = 0;      
    }

  int    charge;
  int    multiplicity;
  double lshift;     //level shift factor, +ve value b/w 0 and 1
  double tol_int;    //tolerance for integral engine
  double tol_lindep; //tolerance for linear dependencies
  double conve;      //energy convergence
  double convd;      //density convergence
  int    diis_hist;  //number of diis history entries
  int    AO_tilesize; 
  bool   restart;    //Read movecs from disk
  bool   noscf;      //only recompute energy from movecs
  bool   ediis;
  double ediis_off;
  bool   sad;
  bool   force_tilesize;
  int    scalapack_nb;
  int    riscf;
  int    nnodes;
  int    scalapack_np_row;
  int    scalapack_np_col;  
  std::string riscf_str;
  std::string moldenfile;
  //ignored when moldenfile not provided
  int n_lindep;
  int writem; 
  double alpha; //density mixing parameter
  std::string scf_type;
  
    void print() {
      std::cout << std::defaultfloat;
      cout << endl << "SCF Options" << endl;
      cout << "{" << endl;
      cout << " charge       = " << charge       << endl;
      cout << " multiplicity = " << multiplicity << endl;
      cout << " level shift  = " << lshift       << endl;
      cout << " tol_int      = " << tol_int      << endl;
      cout << " tol_lindep   = " << tol_lindep   << endl;
      cout << " conve        = " << conve        << endl;
      cout << " convd        = " << convd        << endl;
      cout << " diis_hist    = " << diis_hist    << endl;
      cout << " AO_tilesize  = " << AO_tilesize  << endl;  
      cout << " writem       = " << writem       << endl;  
      if(alpha != 0.7) 
        cout << " alpha        = " << alpha << endl;
      if(!moldenfile.empty()) {
        cout << " moldenfile   = " << moldenfile << endl;    
        //cout << " n_lindep = " << n_lindep << endl;
      }
      if(scalapack_nb>1) 
        cout << " scalapack_nb = " << scalapack_nb << endl;
      if(scalapack_np_row>0) 
        cout << " scalapack_np_row = " << scalapack_np_row << endl;
      if(scalapack_np_col>0) 
        cout << " scalapack_np_col = " << scalapack_np_col << endl;
      print_bool(" restart     ", restart);
      print_bool(" debug       ", debug); 
      if(restart) print_bool(" noscf       ", noscf);
      print_bool(" ediis       ", ediis);
      cout << " ediis_off    = " << ediis_off   << endl;  
      print_bool(" sad         ", sad); 
      cout << "}" << endl;
    }
};

class CDOptions: public Options {

  public:
    CDOptions() = default;
    CDOptions(Options o): Options(o)
    {
      diagtol = 1e-6;
      // At most 8*ao CholVec's. For vast majority cases, this is way
      // more than enough. For very large basis, it can be increased.
      max_cvecs_factor = 12; 
    }

  double diagtol;
  int    max_cvecs_factor;

  void print() {
    std::cout << std::defaultfloat;
    cout << endl << "CD Options" << endl;
    cout << "{" << endl;
    cout << " diagtol          = " << diagtol          << endl;
    cout << " max_cvecs_factor = " << max_cvecs_factor << endl;
    print_bool(" debug           ", debug);   
    cout << "}" << endl; 
  }
};

class CCSDOptions: public Options {
  public:
  CCSDOptions() = default;
  CCSDOptions(Options o): Options(o)
  {
    threshold      = 1e-6;
    tilesize       = 50;
    itilesize      = 1000;
    ccsdt_tilesize = 28;
    ngpu           = 0;
    ndiis          = 5;
    lshift         = 0;
    eom_nroots     = 0;
    eom_threshold  = 1e-6;
    eom_microiter  = o.maxiter;
    writet         = false;
    writet_iter    = ndiis;
    force_tilesize = false;
    readt          = false;
    gf_ip          = true;
    gf_ea          = false;
    gf_os          = false;
    gf_cs          = true;
    gf_restart     = false;
    gf_itriples    = false;
    ccsd_maxiter   = 50;
    balance_tiles  = false;
    profile_ccsd   = false;
    
    gf_p_oi_range        = 0; //1-number of occupied, 2-all MOs
    gf_ndiis             = 10;
    gf_ngmres            = 10;
    gf_maxiter           = 500;
    gf_eta               = -0.01;       
    gf_damping_factor    = 1.0;
    // gf_level_shift    = 0;
    gf_nprocs_poi        = 0;
    // gf_omega          = -0.4; //a.u (range min to max)     
    gf_threshold         = 1e-2;  
    gf_omega_min_ip      = -0.8;  
    gf_omega_max_ip      = -0.4;  
    gf_omega_min_ip_e    = -2.0; 
    gf_omega_max_ip_e    = 0;    
    gf_omega_min_ea      = 0.0;  
    gf_omega_max_ea      = 0.1;  
    gf_omega_min_ea_e    = 0.0; 
    gf_omega_max_ea_e    = 2.0;    
    gf_omega_delta       = 0.01;
    gf_omega_delta_e     = 0.002;
    gf_extrapolate_level = 0;
    gf_analyze_level     = 0;
    gf_analyze_num_omega = 0;
  }

  int    eom_nroots;
  int    tilesize;
  int    itilesize;
  int    ccsdt_tilesize;
  bool   force_tilesize;
  int    ngpu;
  int    ndiis;
  int    eom_microiter;
  int    writet_iter;
  bool   readt, writet, gf_restart, gf_ip, gf_ea, gf_os, gf_cs, 
         gf_itriples, balance_tiles;
  bool   profile_ccsd;
  double lshift;
  double threshold;
  double eom_threshold;
  int    ccsd_maxiter;

  //GF
  int    gf_p_oi_range;
  int    gf_ndiis;
  int    gf_ngmres;  
  int    gf_maxiter;
  double gf_eta;
  // double gf_level_shift;
  int    gf_nprocs_poi;
  double gf_damping_factor;
  // double gf_omega;       
  double gf_threshold;
  double gf_omega_min_ip;
  double gf_omega_max_ip;
  double gf_omega_min_ip_e;
  double gf_omega_max_ip_e;
  double gf_omega_min_ea;
  double gf_omega_max_ea;
  double gf_omega_min_ea_e;
  double gf_omega_max_ea_e;
  double gf_omega_delta;
  double gf_omega_delta_e;
  int    gf_extrapolate_level;
  int    gf_analyze_level;
  int    gf_analyze_num_omega;
  std::vector<double> gf_analyze_omega;
  //Force processing of specified orbitals first
  std::vector<size_t> gf_orbitals;
  
  void print() {
    std::cout << std::defaultfloat;
    cout << endl << "CCSD Options" << endl;
    cout << "{" << endl;
    if(ngpu > 0) {
      cout << " ngpu                 = " << ngpu          << endl;
      cout << " ccsdt_tilesize       = " << ccsdt_tilesize << endl;
    }
    cout << " ndiis                = " << ndiis            << endl;
    cout << " threshold            = " << threshold        << endl;
    cout << " tilesize             = " << tilesize         << endl;
    cout << " ccsd_maxiter         = " << ccsd_maxiter     << endl;
    cout << " itilesize            = " << itilesize        << endl;
    if(lshift != 0) 
      cout << " lshift               = " << lshift           << endl;    
    if(gf_nprocs_poi > 0) 
      cout << " gf_nprocs_poi        = " << gf_nprocs_poi  << endl;
    print_bool(" readt               ", readt); 
    print_bool(" writet              ", writet);
    cout << " writet_iter          = " << writet_iter      << endl;
    print_bool(" profile_ccsd        ", profile_ccsd);
    print_bool(" balance_tiles       ", balance_tiles); 

    if(eom_nroots > 0){
      cout << " eom_nroots           = " << eom_nroots        << endl;
      cout << " eom_microiter        = " << eom_microiter     << endl;
      cout << " eom_threshold        = " << eom_threshold     << endl;
    }

    if(gf_p_oi_range > 0) {
      cout << " gf_p_oi_range        = " << gf_p_oi_range     << endl;
      print_bool(" gf_ip               ", gf_ip); 
      print_bool(" gf_ea               ", gf_ea); 
      print_bool(" gf_os               ", gf_os); 
      print_bool(" gf_cs               ", gf_cs); 
      print_bool(" gf_restart          ", gf_restart);     
      print_bool(" gf_itriples         ", gf_itriples);       
      cout << " gf_ndiis             = " << gf_ndiis          << endl;
      cout << " gf_ngmres            = " << gf_ngmres         << endl;
      cout << " gf_maxiter           = " << gf_maxiter        << endl;
      cout << " gf_eta               = " << gf_eta            << endl;
      // cout << " gf_level_shift         = " << gf_level_shift << endl;
      cout << " gf_damping_factor    = " << gf_damping_factor << endl;
      
      // cout << " gf_omega       = " << gf_omega << endl;
      cout << " gf_threshold         = " << gf_threshold      << endl;
      cout << " gf_omega_min_ip      = " << gf_omega_min_ip   << endl;
      cout << " gf_omega_max_ip      = " << gf_omega_max_ip   << endl;
      cout << " gf_omega_min_ip_e    = " << gf_omega_min_ip_e << endl;
      cout << " gf_omega_max_ip_e    = " << gf_omega_max_ip_e << endl;
      cout << " gf_omega_min_ea      = " << gf_omega_min_ea   << endl;
      cout << " gf_omega_max_ea      = " << gf_omega_max_ea   << endl;
      cout << " gf_omega_min_ea_e    = " << gf_omega_min_ea_e << endl;
      cout << " gf_omega_max_ea_e    = " << gf_omega_max_ea_e << endl;
      cout << " gf_omega_delta       = " << gf_omega_delta    << endl; 
      cout << " gf_omega_delta_e     = " << gf_omega_delta_e  << endl; 
      if(!gf_orbitals.empty()) {
        cout << " gf_orbitals     = [";
        for(auto x: gf_orbitals) cout << x << ",";
        cout << "]" << endl;           
      }
      if(gf_analyze_level > 0) {
        cout << " gf_analyze_level     = " << gf_analyze_level     << endl; 
        cout << " gf_analyze_num_omega = " << gf_analyze_num_omega << endl; 
        cout << " gf_analyze_omega     = [";
        for(auto x: gf_analyze_omega) cout << x << ",";
        cout << "]" << endl;      
      }
      if(gf_extrapolate_level>0) cout << " gf_extrapolate_level = " << gf_extrapolate_level << endl; 
    }   

    print_bool(" debug               ", debug); 
    cout << "}" << endl;
  }

};

class OptionsMap
{
  public:
    OptionsMap() = default;
    Options options;
    SCFOptions scf_options;
    CDOptions cd_options;
    CCSDOptions ccsd_options;
};


void nwx_terminate(std::string msg) {
    if(GA_Nodeid()==0) std::cout << msg << " ... terminating program." << endl << endl;
    GA_Terminate();
    MPI_Finalize();
    exit(0);
}

void to_upper(std::string& str) { std::transform(str.begin(), str.end(), str.begin(), ::toupper); }
void to_lower(std::string& str) { std::transform(str.begin(), str.end(), str.begin(), ::tolower); }

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

bool to_bool(const std::string str) { 
  if (is_in_line("false",str)) return false;
  return true;
}

bool is_empty(std::string line){
  if(line.find_first_not_of(' ') == std::string::npos 
    || line.empty() || is_comment(line)) return true;
  return false;
}

void skip_empty_lines(std::istream& is) {
    std::string line;
    auto curpos = is.tellg();
    std::getline(is, line);
    
    if(is_empty(line)) {
      curpos = is.tellg();
      auto trackpos = curpos;
      while (is_empty(line)) {
        curpos = trackpos;
        std::getline(is, line);
        trackpos = is.tellg();
      }
    }
    is.clear();//cannot seek to curpos if eof is reached
    is.seekg(curpos,std::ios_base::beg);
    // std::getline(is, line);
}


void check_start(std::string line, std::string section) {
    auto spos = line.find("{");
    std::string rest = line.substr(spos+1,line.length());
    if(!is_empty(rest)) {
      nwx_terminate("ERROR in " + section + " section: " \
      + "cannot have any option in the same line as \"{\" ");
    }
}

void check_section_start(std::istream& is, std::string line, std::string section) {
    skip_empty_lines(is);
    //Check opening curly brace
    if(!is_in_line("{",line)) {
        std::getline(is, line);
        if(!is_in_line("{",line))
          nwx_terminate("missing { in " + section + " section");
        else check_start(line,section);
    }
    else check_start(line,section);
}

void unknown_option(const std::string line, const std::string section){
  if(!is_comment(line)) {
    std::cout << "ERROR: unknown " + section + " option: " << line << endl;
    nwx_terminate("Remove/comment above line from the input file.\
 This can also happen when missing \"}\" to close the " + section + " section.");
  }
}


std::vector<Atom> read_atoms(std::istream& is) {
    
    // first line = # of atoms
    skip_empty_lines(is);
    size_t natom;
    is >> natom;
    // read off the rest of first line and discard
    std::string rest_of_line;
    std::getline(is, rest_of_line);
    if(rest_of_line.length()>0)
      cerr << "Ignoring unsupported option: " << rest_of_line << endl;

    skip_empty_lines(is);

    // rest of lines are atoms
    std::vector<Atom> atoms(natom);
    for(size_t i = 0; i < natom; i++) {
        // read line
        std::string line;
        std::getline(is, line);
        std::istringstream iss(line);
        // then parse ... this handles "extended" XYZ formats
        std::string element_symbol;
        double x, y, z;
        iss >> element_symbol >> x >> y >> z;

        // .xyz files report element labels, hence convert to atomic numbers
        int Z = -1;
        for(const auto& e : libint2::chemistry::get_element_info()) {
            if(strequal_case(e.symbol, element_symbol)) {
                Z = e.Z;
                break;
            }
        }
        if(Z == -1) {
            std::ostringstream oss;
            oss << "read_dotxyz: element symbol \"" << element_symbol
                << "\" is not recognized" << std::endl;
            throw std::runtime_error(oss.str().c_str());
        }

        atoms[i].atomic_number = Z;

        atoms[i].x = x;
        atoms[i].y = y;
        atoms[i].z = z;

    }

    return atoms;
}

std::tuple<Options, SCFOptions, CDOptions, CCSDOptions> read_nwx_file(std::istream& is) {
      //Can have blank/comment lines after atom list
    skip_empty_lines(is);

    std::string line;
    bool section_start = true;

    //Parsing common options
    Options options;
    std::getline(is, line);
    //1st non-empty line after atom list should be common section
    if(!is_in_line("COMMON", line)) 
      nwx_terminate("COMMON section missing/incorrectly placed in input file");
    
    check_section_start(is, line, "COMMON");

    while(section_start){
      skip_empty_lines(is);
      std::getline(is, line);

      if(is_in_line("basis",line)) {
        std::istringstream iss(line);
        std::vector<std::string> basis_line{std::istream_iterator<std::string>{iss},
                                            std::istream_iterator<std::string>{}};
        assert(basis_line.size() == 2 || basis_line.size()==3);
        options.basis = basis_line[1];
        if(basis_line.size()==3){
          options.sphcart = basis_line[2];
          if(!strequal_case(options.sphcart, "cartesian") && !strequal_case(options.sphcart, "spherical"))
            nwx_terminate("unknown sphcart value for basis specified");      
        }
      }
      else if(is_in_line("maxiter",line))
        options.maxiter = std::stoi(read_option(line));
      else if(is_in_line("debug",line))
        options.debug = to_bool(read_option(line));        
      else if(is_in_line("dfbasis",line)) 
        options.dfbasis = read_option(line);  
      else if(is_in_line("output_file_prefix",line)) 
        options.output_file_prefix = read_option(line);          
      else if(is_in_line("geometry",line)){
        //geometry units
        std::istringstream iss(line);
        std::vector<std::string> geom_units{std::istream_iterator<std::string>{iss},
                                            std::istream_iterator<std::string>{}};
        assert(geom_units.size() == 3);
        auto gunit = geom_units[2];
        if(!strequal_case(gunit,"bohr") && !strequal_case(gunit,"angstrom"))
          nwx_terminate("unknown geometry units specified");
        options.geom_units = gunit;
      }
      else if(is_in_line("}",line)) section_start = false;
      else unknown_option(line,"");
    }
 
    SCFOptions scf_options(options);
    CDOptions cd_options(options);
    CCSDOptions ccsd_options(options);

    skip_empty_lines(is);

    while(!is.eof()) {
      std::getline(is, line);

      if(is_in_line("SCF",line)){
        section_start = true;
        check_section_start(is, line, "SCF");

        while(section_start){
          skip_empty_lines(is);
          std::getline(is, line);

          if(is_in_line("charge",line)) 
            scf_options.charge = std::stod(read_option(line));   
          else if(is_in_line("multiplicity",line)) 
            scf_options.multiplicity = std::stod(read_option(line));   
          else if(is_in_line("lshift",line)) 
            scf_options.lshift = std::stod(read_option(line));              
          else if(is_in_line("tol_int",line)) 
            scf_options.tol_int = std::stod(read_option(line));   
          else if(is_in_line("tol_lindep",line)) 
            scf_options.tol_lindep = std::stod(read_option(line));
          else if(is_in_line("conve",line)) 
            scf_options.conve = std::stod(read_option(line));
          else if(is_in_line("convd",line)) 
            scf_options.convd = std::stod(read_option(line));            
          else if(is_in_line("diis_hist",line)) 
            scf_options.diis_hist = std::stoi(read_option(line));    
          else if(is_in_line("force_tilesize",line)) 
            scf_options.force_tilesize = to_bool(read_option(line));  
          else if(is_in_line("tilesize",line)) 
            scf_options.AO_tilesize = std::stod(read_option(line)); 
          else if(is_in_line("alpha",line)) 
            scf_options.alpha = std::stod(read_option(line));  
          else if(is_in_line("writem",line)) 
            scf_options.writem = std::stoi(read_option(line));    
          else if(is_in_line("nnodes",line)) 
            scf_options.nnodes = std::stoi(read_option(line));                                      
          else if(is_in_line("riscf",line)) {
            std::string riscf_str = read_option(line);
            if(riscf_str == "J") scf_options.riscf = 1;
            else if(riscf_str == "K") scf_options.riscf = 2;
          }
          else if(is_in_line("restart",line))
            scf_options.restart = to_bool(read_option(line));  
          else if(is_in_line("noscf",line))
            scf_options.noscf = to_bool(read_option(line));     
          else if(is_in_line("ediis",line))
            scf_options.ediis = to_bool(read_option(line));
          else if(is_in_line("ediis_off",line))
            scf_options.ediis_off = std::stod(read_option(line));  
          else if(is_in_line("sad",line))
            scf_options.sad = to_bool(read_option(line));                  
          else if(is_in_line("debug",line))
            scf_options.debug = to_bool(read_option(line)); 
          else if(is_in_line("moldenfile",line))
            scf_options.moldenfile = read_option(line); 
          else if(is_in_line("scf_type",line))
            scf_options.scf_type = read_option(line); 
          else if(is_in_line("n_lindep",line))
            scf_options.n_lindep = std::stoi(read_option(line)); 
          else if(is_in_line("scalapack_nb",line)) 
            scf_options.scalapack_nb = std::stoi(read_option(line));   
          else if(is_in_line("scalapack_np_row",line)) 
            scf_options.scalapack_np_row = std::stoi(read_option(line));                                                             
          else if(is_in_line("scalapack_np_col",line)) 
            scf_options.scalapack_np_col = std::stoi(read_option(line));               
          else if(is_in_line("}",line)) section_start = false;
          else unknown_option(line,"SCF");
          
        }
      }

      else if(is_in_line("CD", line)) {
        section_start = true;
        check_section_start(is, line, "CD");

        while(section_start){
          skip_empty_lines(is);
          std::getline(is, line);

          if(is_in_line("max_cvecs",line)) 
            cd_options.max_cvecs_factor = std::stoi(read_option(line));    
          else if(is_in_line("diagtol",line)) 
            cd_options.diagtol = std::stod(read_option(line));  
          else if(is_in_line("debug",line))
            cd_options.debug = to_bool(read_option(line));                               
          else if(is_in_line("}",line)) section_start = false;
          else unknown_option(line, "CD");
        }
      }

      else if (is_in_line("CCSD", line)) {
        section_start = true;
        check_section_start(is, line, "CCSD");

        while(section_start){
          skip_empty_lines(is);
          std::getline(is, line);

          if(is_in_line("ndiis",line)) 
            ccsd_options.ndiis = std::stoi(read_option(line));  
          else if(is_in_line("eom_nroots",line)) 
            ccsd_options.eom_nroots = std::stoi(read_option(line));  
          else if(is_in_line("eom_microiter",line)) 
            ccsd_options.eom_microiter = std::stoi(read_option(line));  
          else if(is_in_line("ccsd_maxiter",line)) 
            ccsd_options.ccsd_maxiter = std::stoi(read_option(line)); 
          else if(is_in_line("lshift",line)) 
            ccsd_options.lshift = std::stod(read_option(line));                                      
          else if(is_in_line("eom_threshold",line)) 
            ccsd_options.eom_threshold = std::stod(read_option(line));              
          else if(is_in_line("threshold",line)) 
            ccsd_options.threshold = std::stod(read_option(line));  
          else if(is_in_line("tilesize",line))
            ccsd_options.tilesize = std::stoi(read_option(line));
          else if(is_in_line("ccsdt_tilesize",line))
            ccsd_options.ccsdt_tilesize = std::stoi(read_option(line));            
          else if(is_in_line("itilesize",line))
            ccsd_options.itilesize = std::stoi(read_option(line));            
          else if(is_in_line("ngpu",line))
            ccsd_options.ngpu = std::stoi(read_option(line));            
          else if(is_in_line("debug",line))
            ccsd_options.debug = to_bool(read_option(line)); 
          else if(is_in_line("readt",line))
            ccsd_options.readt = to_bool(read_option(line)); 
          else if(is_in_line("writet",line))
            ccsd_options.writet = to_bool(read_option(line));
          else if(is_in_line("writet_iter",line))
            ccsd_options.writet_iter = std::stoi(read_option(line));            
          else if(is_in_line("balance_tiles",line))
            ccsd_options.balance_tiles = to_bool(read_option(line)); 
          else if(is_in_line("profile_ccsd",line))
            ccsd_options.profile_ccsd = to_bool(read_option(line));                 
          else if(is_in_line("force_tilesize",line)) 
            ccsd_options.force_tilesize = to_bool(read_option(line));                     
          else if(is_in_line("gf_ip",line))
            ccsd_options.gf_ip = to_bool(read_option(line)); 
          else if(is_in_line("gf_ea",line))
            ccsd_options.gf_ea = to_bool(read_option(line)); 
          else if(is_in_line("gf_os",line))
            ccsd_options.gf_os = to_bool(read_option(line)); 
          else if(is_in_line("gf_cs",line))
            ccsd_options.gf_cs = to_bool(read_option(line)); 
          else if(is_in_line("gf_restart",line))
            ccsd_options.gf_restart = to_bool(read_option(line)); 
          else if(is_in_line("gf_itriples",line))
            ccsd_options.gf_itriples = to_bool(read_option(line));             
          else if(is_in_line("gf_p_oi_range",line)) {
            ccsd_options.gf_p_oi_range = std::stoi(read_option(line)); 
            if(ccsd_options.gf_p_oi_range != 1 && ccsd_options.gf_p_oi_range != 2)
              nwx_terminate ("gf_p_oi_range can only be one of 1 or 2");
          }
          else if(is_in_line("gf_ndiis",line)) 
            ccsd_options.gf_ndiis = std::stoi(read_option(line)); 
          else if(is_in_line("gf_ngmres",line)) 
            ccsd_options.gf_ngmres = std::stoi(read_option(line)); 
          else if(is_in_line("gf_maxiter",line)) 
            ccsd_options.gf_maxiter = std::stoi(read_option(line));
          else if(is_in_line("gf_nprocs_poi",line)) 
            ccsd_options.gf_nprocs_poi = std::stoi(read_option(line));                         
          // else if(is_in_line("gf_level_shift",line)) 
          //   ccsd_options.gf_level_shift = std::stod(read_option(line));  
          else if(is_in_line("gf_damping_factor",line)) 
            ccsd_options.gf_damping_factor = std::stod(read_option(line));              
          else if(is_in_line("gf_eta",line)) 
            ccsd_options.gf_eta = std::stod(read_option(line));                     
          else if(is_in_line("gf_threshold",line)) 
            ccsd_options.gf_threshold = std::stod(read_option(line));  
          else if(is_in_line("gf_omega_min_ip",line)) 
            ccsd_options.gf_omega_min_ip = std::stod(read_option(line));  
          else if(is_in_line("gf_omega_max_ip",line)) 
            ccsd_options.gf_omega_max_ip = std::stod(read_option(line));  
          else if(is_in_line("gf_omega_min_ip_e",line)) 
            ccsd_options.gf_omega_min_ip_e = std::stod(read_option(line));  
          else if(is_in_line("gf_omega_max_ip_e",line)) 
            ccsd_options.gf_omega_max_ip_e = std::stod(read_option(line));  
          else if(is_in_line("gf_omega_min_ea",line)) 
            ccsd_options.gf_omega_min_ea = std::stod(read_option(line));  
          else if(is_in_line("gf_omega_max_ea",line)) 
            ccsd_options.gf_omega_max_ea = std::stod(read_option(line));  
          else if(is_in_line("gf_omega_min_ea_e",line)) 
            ccsd_options.gf_omega_min_ea_e = std::stod(read_option(line));  
          else if(is_in_line("gf_omega_max_ea_e",line)) 
            ccsd_options.gf_omega_max_ea_e = std::stod(read_option(line));  
          else if(is_in_line("gf_omega_delta",line)) 
            ccsd_options.gf_omega_delta = std::stod(read_option(line));  
          else if(is_in_line("gf_omega_delta_e",line)) 
            ccsd_options.gf_omega_delta_e = std::stod(read_option(line));              
          else if(is_in_line("gf_extrapolate_level",line)) 
            ccsd_options.gf_extrapolate_level = std::stoi(read_option(line)); 
          else if(is_in_line("gf_analyze_level",line)) 
            ccsd_options.gf_analyze_level = std::stoi(read_option(line));  
          else if(is_in_line("gf_analyze_num_omega",line)) 
            ccsd_options.gf_analyze_num_omega = std::stoi(read_option(line));              
          else if(is_in_line("gf_analyze_omega",line)) {
              std::istringstream iss(line);
              std::string wignore;
              iss >> wignore;
              std::vector<double> gf_analyze_omega{std::istream_iterator<double>{iss},
                                              std::istream_iterator<double>{}};
              ccsd_options.gf_analyze_omega = gf_analyze_omega;
          }          
          else if(is_in_line("gf_orbitals",line)) {
              std::istringstream iss(line);
              std::string wignore;
              iss >> wignore;
              std::vector<size_t> gf_orbitals{std::istream_iterator<double>{iss},
                                              std::istream_iterator<double>{}};
              ccsd_options.gf_orbitals = gf_orbitals;
          }                    
          else if(is_in_line("}",line)) section_start = false;
          else unknown_option(line, "CCSD");

        }
      }
      //else ignore 
    }

    return std::make_tuple(options, scf_options, cd_options, ccsd_options);

}


inline std::tuple<std::vector<Atom>, OptionsMap>
   read_input_nwx(std::istream& is) {

    const double angstrom_to_bohr =
      1.889725989; // 1 / bohr_to_angstrom; //1.889726125
    
    auto atoms = read_atoms(is);

    auto [options, scf_options, cd_options, ccsd_options] = read_nwx_file(is);

    //Done parsing input file
    {
      //If geometry units specified are angstrom, convert to bohr
      bool nw_units_bohr = true;
      if(options.geom_units == "angstrom") nw_units_bohr = false;

      if(!nw_units_bohr){
        // .xyz files report Cartesian coordinates in angstroms; 
        // convert to bohr
        for(auto i = 0U; i < atoms.size(); i++){
          atoms[i].x *= angstrom_to_bohr;
          atoms[i].y *= angstrom_to_bohr;
          atoms[i].z *= angstrom_to_bohr;
        }
      }
    }

    if(GA_Nodeid()==0){
      options.print();
      // scf_options.print();
      // cd_options.print();
      // ccsd_options.print();
    }

    OptionsMap options_map;
    options_map.options = options;
    options_map.scf_options = scf_options;
    options_map.cd_options = cd_options;

    if(ccsd_options.eom_microiter < ccsd_options.maxiter &&
       ccsd_options.eom_microiter == options.maxiter) 
       ccsd_options.eom_microiter = ccsd_options.maxiter;

    options_map.ccsd_options = ccsd_options;

    return std::make_tuple(atoms, options_map);
}


#endif // TAMM_METHODS_INPUT_PARSER_HPP_
