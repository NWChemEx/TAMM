
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
    }

  double tol_int; //tolerance for integral engine
  double tol_lindep; //tolerance for linear dependencies
  double conve; //energy convergence
  double convd; //density convergence
  int diis_hist; //number of diis history entries
  int AO_tilesize; 
  bool restart; //Read orbitals from disk

    void print() {
      cout << "\nSCF Options\n";
      cout << "{\n";
      cout << " tol_int = " << tol_int << endl;
      cout << " tol_lindep = " << tol_lindep << endl;
      cout << " conve = " << conve << endl;
      cout << " convd = " << convd << endl;
      cout << " diis_hist = " << diis_hist << endl;
      cout << " AO_tilesize = " << AO_tilesize << endl;      
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
      max_cvecs_factor = 8; 
    }

  double diagtol;
  int max_cvecs_factor;

  void print() {
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
    tilesize = 30;
    eom_nroots = 1;
    eom_threshold = 1e-10;
    eom_microiter = o.maxiter;
  }

  int eom_nroots;
  int tilesize;
  int eom_microiter;
  double threshold;
  double eom_threshold;

  void print() {
    cout << "\nCCSD Options\n";
    cout << "{\n";
    cout << " threshold = " << threshold << endl;
    cout << " tilesize = " << tilesize << endl;
    cout << " eom_nroots = " << eom_nroots << endl;
    cout << " eom_microiter = " << eom_microiter << endl;
    cout << " eom_threshold = " << eom_threshold << endl;
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
    std::cerr << msg << " ... terminating program.\n";
    GA_Terminate();
    MPI_Finalize();
    exit(1);
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
  else if (line.find(str_u) == std::string::npos &&
      line.find(str_l) == std::string::npos) found = false;

  //TODO
  if (str_l == "basis"){
    if(line.find("dfbasis") != std::string::npos || 
       line.find("DFBASIS") != std::string::npos) found=false;
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
            if(libint2::strcaseequal(e.symbol, element_symbol)) {
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
          else if(is_in_line("tilesize",line)) 
            scf_options.AO_tilesize = std::stod(read_option(line));  
          else if(is_in_line("restart",line))
            scf_options.restart = to_bool(read_option(line));        
          else if(is_in_line("debug",line))
            scf_options.debug = to_bool(read_option(line));                                           
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
          else if(is_in_line("eom_threshold",line)) 
            ccsd_options.eom_threshold = std::stod(read_option(line));              
          else if(is_in_line("threshold",line)) 
            ccsd_options.threshold = std::stod(read_option(line));  
          else if(is_in_line("tilesize",line))
            ccsd_options.tilesize = std::stoi(read_option(line));
          else if(is_in_line("debug",line))
            ccsd_options.debug = to_bool(read_option(line));                               
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
