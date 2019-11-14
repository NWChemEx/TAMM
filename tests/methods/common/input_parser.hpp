
#ifndef TAMM_TESTS_INPUT_PARSER_HPP_
#define TAMM_TESTS_INPUT_PARSER_HPP_

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
    }

    bool debug;
    int maxiter;
    std::string basis;
    std::string dfbasis;
    std::string geom_units;

    void print() {
      std::cout << std::defaultfloat;
      cout << "\nCommon Options\n";
      cout << "{\n";
      cout << " max iter = " << maxiter << endl;
      cout << " basis = " << basis << endl;
      if(!dfbasis.empty()) cout << " dfbasis = " << dfbasis << endl;
      cout << " geom_units = " << geom_units << endl;
      print_bool(" debug", debug);
      cout << "}\n";
    }
};

class SCFOptions: public Options {

  public:
    SCFOptions() = default;
    SCFOptions(Options o): Options(o)
    {
      tol_int = 1e-8;
      tol_lindep = 1e-6;
      conve = 1e-6;
      convd = 1e-5;
      diis_hist = 10;
      AO_tilesize = 30;
      restart = false;
      scalapack_nb = 1;
      scalapack_np_row = 0;
      scalapack_np_col = 0;
      force_tilesize = false;
      riscf = 0; //0 for JK, 1 for J, 2 for K
      riscf_str = "JK";
    }

  double tol_int; //tolerance for integral engine
  double tol_lindep; //tolerance for linear dependencies
  double conve; //energy convergence
  double convd; //density convergence
  int diis_hist; //number of diis history entries
  int AO_tilesize; 
  bool restart; //Read movecs from disk
  bool force_tilesize;
  int scalapack_nb;
  int scalapack_np_row;
  int scalapack_np_col;
  int riscf;
  std::string riscf_str;

    void print() {
      std::cout << std::defaultfloat;
      cout << "\nSCF Options\n";
      cout << "{\n";
      cout << " tol_int = " << tol_int << endl;
      cout << " tol_lindep = " << tol_lindep << endl;
      cout << " conve = " << conve << endl;
      cout << " convd = " << convd << endl;
      cout << " diis_hist = " << diis_hist << endl;
      cout << " AO_tilesize = " << AO_tilesize << endl;  
      cout << " riscf = " << riscf_str << endl;     
      if(scalapack_nb>1) cout << " scalapack_nb = " << scalapack_nb << endl;
      if(scalapack_np_row>0) cout << " scalapack_np_row = " << scalapack_np_row << endl;
      if(scalapack_np_col>0) cout << " scalapack_np_col = " << scalapack_np_col << endl;
      print_bool(" restart", restart);
      print_bool(" debug", debug); 
      cout << "}\n";
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
  int max_cvecs_factor;

  void print() {
    std::cout << std::defaultfloat;
    cout << "\nCD Options\n";
    cout << "{\n";
    cout << " diagtol = " << diagtol << endl;
    cout << " max_cvecs_factor = " << max_cvecs_factor << endl;
    print_bool(" debug", debug);   
    cout << "}\n"; 
  }
};

class CCSDOptions: public Options {
  public:
  CCSDOptions() = default;
  CCSDOptions(Options o): Options(o)
  {
    threshold = 1e-10;
    tilesize = 50;
    itilesize = 1000;
    icuda = 0;
    eom_nroots = 0;
    eom_threshold = 1e-10;
    eom_microiter = o.maxiter;
    writet = false;
    readt = false;
    gf_restart = true;
    ccsd_maxiter = 50;
    
    gf_p_oi_range = 0; //1-number of occupied, 2-all MOs
    gf_ndiis = 10;
    gf_maxiter = 500;
    gf_eta = -0.01;       
    gf_damping_factor = 1.0;
    // gf_level_shift = 0;
    gf_nprocs_poi = 0;
    // gf_omega = -0.4; //a.u (range min to max)     
    gf_threshold = 1e-2;  
    gf_omega_min = -0.8;  
    gf_omega_max = -0.4;  
    gf_omega_min_e = -2.0; 
    gf_omega_max_e = 0;    
    gf_omega_delta = 0.01;
    gf_omega_delta_e = 0.002;
    gf_extrapolate_level = 0;

    gf_analyze_level = 0;
    gf_analyze_num_omega = 0;
  }

  int eom_nroots;
  int tilesize;
  int itilesize;
  int icuda;
  int eom_microiter;
  bool readt, writet, gf_restart;
  double threshold;
  double eom_threshold;
  int ccsd_maxiter;

  //GF
  int gf_p_oi_range;
  int gf_ndiis;
  int gf_maxiter;
  double gf_eta;
  // double gf_level_shift;
  int gf_nprocs_poi;
  double gf_damping_factor;
  // double gf_omega;       
  double gf_threshold;
  double gf_omega_min;
  double gf_omega_max;
  double gf_omega_min_e;
  double gf_omega_max_e;
  double gf_omega_delta;
  double gf_omega_delta_e;
  int gf_extrapolate_level;

  int gf_analyze_level;
  int gf_analyze_num_omega;
  std::vector<double> gf_analyze_omega;
  
  void print() {
    std::cout << std::defaultfloat;
    cout << "\nCCSD Options\n";
    cout << "{\n";
    if(icuda > 0) cout << " #cuda = " << icuda << endl;
    cout << " threshold = " << threshold << endl;
    cout << " tilesize = " << tilesize << endl;
    cout << " ccsd_maxiter = " << ccsd_maxiter << endl;
    cout << " itilesize = " << itilesize << endl;
    if(gf_nprocs_poi > 0) cout << " gf_nprocs_poi = " << gf_nprocs_poi << endl;
    print_bool(" readt", readt); 
    print_bool(" writet", writet); 
    print_bool(" gf_restart", gf_restart); 

    if(eom_nroots > 0){
      cout << " eom_nroots = " << eom_nroots << endl;
      cout << " eom_microiter = " << eom_microiter << endl;
      cout << " eom_threshold = " << eom_threshold << endl;
    }

    if(gf_p_oi_range > 0) {
      cout << " gf_p_oi_range  = " << gf_p_oi_range << endl;
      cout << " gf_ndiis       = " << gf_ndiis << endl;
      cout << " gf_maxiter     = " << gf_maxiter << endl;
      cout << " gf_eta         = " << gf_eta << endl;
      // cout << " gf_level_shift         = " << gf_level_shift << endl;
      cout << " gf_damping_factor         = " << gf_damping_factor << endl;
      
      // cout << " gf_omega       = " << gf_omega << endl;
      cout << " gf_threshold   = " << gf_threshold  << endl;
      cout << " gf_omega_min   = " << gf_omega_min  << endl;
      cout << " gf_omega_max   = " << gf_omega_max  << endl;
      cout << " gf_omega_min_e = " << gf_omega_min_e << endl;
      cout << " gf_omega_max_e = " << gf_omega_max_e << endl;
      cout << " gf_omega_delta = " << gf_omega_delta << endl; 
      cout << " gf_omega_delta_e = " << gf_omega_delta_e << endl; 
      if(gf_analyze_level > 0) {
        cout << " gf_analyze_level = " << gf_analyze_level << endl; 
        cout << " gf_analyze_num_omega = " << gf_analyze_num_omega << endl; 
        cout << " gf_analyze_omega = [";
        for(auto x: gf_analyze_omega) cout << x << ",";
        cout << "]" << endl;      
      }
      if(gf_extrapolate_level>0) cout << " gf_extrapolate_level = " << gf_extrapolate_level << endl; 
    }   

    print_bool(" debug", debug); 
    cout << "}\n";
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


void nwx_terminate(std::string msg){
    if(GA_Nodeid()==0) std::cout << msg << " ... terminating program.\n\n";
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

      if(is_in_line("basis",line)) 
        options.basis = read_option(line);      
      else if(is_in_line("maxiter",line))
        options.maxiter = std::stoi(read_option(line));
      else if(is_in_line("debug",line))
        options.debug = to_bool(read_option(line));        
      else if(is_in_line("dfbasis",line)) 
        options.dfbasis = read_option(line);    
      else if(is_in_line("geometry",line)){
        //geometry units
        std::istringstream iss(line);
        std::vector<std::string> geom_units{std::istream_iterator<std::string>{iss},
                                            std::istream_iterator<std::string>{}};
        assert(geom_units.size() == 3);
        auto gunit = geom_units[2];
        if(gunit != "bohr" && gunit != "angstrom")
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

          if(is_in_line("tol_int",line)) 
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
          else if(is_in_line("riscf",line)) {
            std::string riscf_str = read_option(line);
            if(riscf_str == "J") scf_options.riscf = 1;
            else if(riscf_str == "K") scf_options.riscf = 2;
          }
          else if(is_in_line("restart",line))
            scf_options.restart = to_bool(read_option(line));        
          else if(is_in_line("debug",line))
            scf_options.debug = to_bool(read_option(line));         
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

          if(is_in_line("eom_nroots",line)) 
            ccsd_options.eom_nroots = std::stoi(read_option(line));  
          else if(is_in_line("eom_microiter",line)) 
            ccsd_options.eom_microiter = std::stoi(read_option(line));  
          else if(is_in_line("ccsd_maxiter",line)) 
            ccsd_options.ccsd_maxiter = std::stoi(read_option(line));                          
          else if(is_in_line("eom_threshold",line)) 
            ccsd_options.eom_threshold = std::stod(read_option(line));              
          else if(is_in_line("threshold",line)) 
            ccsd_options.threshold = std::stod(read_option(line));  
          else if(is_in_line("tilesize",line))
            ccsd_options.tilesize = std::stoi(read_option(line));
          else if(is_in_line("itilesize",line))
            ccsd_options.itilesize = std::stoi(read_option(line));            
          else if(is_in_line("cuda",line))
            ccsd_options.icuda = std::stoi(read_option(line));            
          else if(is_in_line("debug",line))
            ccsd_options.debug = to_bool(read_option(line)); 
          else if(is_in_line("readt",line))
            ccsd_options.readt = to_bool(read_option(line)); 
          else if(is_in_line("writet",line))
            ccsd_options.writet = to_bool(read_option(line));    
          else if(is_in_line("gf_restart",line))
            ccsd_options.gf_restart = to_bool(read_option(line)); 
          else if(is_in_line("gf_p_oi_range",line)) {
            ccsd_options.gf_p_oi_range = std::stoi(read_option(line)); 
            if(ccsd_options.gf_p_oi_range != 1 && ccsd_options.gf_p_oi_range != 2)
              nwx_terminate ("gf_p_oi_range can only be one of 1 or 2");
          }
          else if(is_in_line("gf_ndiis",line)) 
            ccsd_options.gf_ndiis = std::stoi(read_option(line)); 
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
          else if(is_in_line("gf_omega_min",line)) 
            ccsd_options.gf_omega_min = std::stod(read_option(line));  
          else if(is_in_line("gf_omega_max",line)) 
            ccsd_options.gf_omega_max = std::stod(read_option(line));  
          else if(is_in_line("gf_omega_min_e",line)) 
            ccsd_options.gf_omega_min_e = std::stod(read_option(line));  
          else if(is_in_line("gf_omega_max_e",line)) 
            ccsd_options.gf_omega_max_e = std::stod(read_option(line));  
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


#endif // TAMM_TESTS_INPUT_PARSER_HPP_
