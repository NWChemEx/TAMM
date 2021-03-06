
#ifndef TAMM_METHODS_INPUT_PARSER_HPP_
#define TAMM_METHODS_INPUT_PARSER_HPP_

#include <cctype>
#include <string>
#include <vector>
#include <iostream>
#include <regex>


// Libint Gaussian integrals library
#include <libint2.hpp>
#include <libint2/basis.h>
#include <libint2/chemistry/sto3g_atomic_density.h>

#include "ga.h"
#include "ga-mpi.h"

#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

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
    std::vector<libint2::Atom> atoms;

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
    printtol       = 0.05;
    threshold      = 1e-6;
    force_tilesize = false;
    tilesize       = 50;
    itilesize      = 1000;
    ndiis          = 5;
    lshift         = 0;
    ccsd_maxiter   = 50;
    balance_tiles  = true;
    profile_ccsd   = false;

    writet         = false;
    writet_iter    = ndiis;
    readt          = false;

    localize       = false;
    skip_dlpno     = false;
    keep_npairs    = 1;
    max_pnos       = 1;
    dlpno_dfbasis  = "";
    tcutpno        = 0;

    ngpu           = 0;
    ccsdt_tilesize = 28;

    eom_nroots     = 0;
    eom_threshold  = 1e-6;
    eom_microiter  = o.maxiter;

    gf_ip          = true;
    gf_ea          = false;
    gf_os          = false;
    gf_cs          = true;
    gf_restart     = false;
    gf_itriples    = false;

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

  int    tilesize;
  int    itilesize;
  bool   force_tilesize;
  int    ndiis;
  int    writet_iter;
  bool   readt, writet, gf_restart, gf_ip, gf_ea, gf_os, gf_cs, 
         gf_itriples, balance_tiles;
  bool   profile_ccsd;
  double lshift;
  double printtol;
  double threshold;

  int    ccsd_maxiter;
  std::string ext_data_path;

  //CCSD(T)
  int    ngpu;
  int    ccsdt_tilesize;

  //DLPNO
  bool   localize;
  bool   skip_dlpno;
  int    max_pnos;
  size_t keep_npairs;
  std::string dlpno_dfbasis;
  double tcutpno;

  //EOM
  int    eom_nroots;
  int    eom_microiter;
  double eom_threshold;

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
    cout << " printtol             = " << printtol         << endl;
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
    
    if(!dlpno_dfbasis.empty()) cout << " dlpno_dfbasis        = " << dlpno_dfbasis << endl; 

    if(!ext_data_path.empty()) {
      cout << " ext_data_path   = " << ext_data_path << endl;    
    }

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
        cout << " gf_orbitals        = [";
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

void tamm_terminate(std::string msg) {
    if(GA_Nodeid()==0) std::cout << msg << " ... terminating program." << endl << endl;
    GA_Terminate();
    MPI_Finalize();
    exit(0);
}

void to_upper(std::string& str) { std::transform(str.begin(), str.end(), str.begin(), ::toupper); }
void to_lower(std::string& str) { std::transform(str.begin(), str.end(), str.begin(), ::tolower); }

template<typename T>
void parse_option(T& val, json j, string key, bool optional=true)
{
  if (j.contains(key)) val = j[key].get<T>();
  else if(!optional) {
    tamm_terminate("ERROR: " + key + " not specified. Please specify the " + key + " option!");
  }
}


std::tuple<Options, SCFOptions, CDOptions, CCSDOptions> parse_json(json& jinput) {

    Options options;
    parse_option<string>(options.basis             , jinput["basis"]   , "basisset",false);
    parse_option<string>(options.sphcart           , jinput["basis"]   , "sphcart");
    parse_option<int>   (options.maxiter           , jinput["common"]  , "maxiter");
    parse_option<bool>  (options.debug             , jinput["common"]  , "debug");
    parse_option<string>(options.dfbasis           , jinput["basis"]   , "df_basisset");
    parse_option<string>(options.geom_units        , jinput["geometry"], "units");
    parse_option<string>(options.output_file_prefix, jinput["common"]  , "output_file_prefix");

    SCFOptions  scf_options(options);
    CDOptions   cd_options(options);
    CCSDOptions ccsd_options(options);

    //SCF
    json jscf = jinput["SCF"];
    parse_option<int>   (scf_options.charge          , jscf, "charge");
    parse_option<int>   (scf_options.multiplicity    , jscf, "multiplicity");   
    parse_option<double>(scf_options.lshift          , jscf, "lshift");
    parse_option<double>(scf_options.tol_int         , jscf, "tol_int");   
    parse_option<double>(scf_options.tol_lindep      , jscf, "tol_lindep");
    parse_option<double>(scf_options.conve           , jscf, "conve");
    parse_option<double>(scf_options.convd           , jscf, "convd");            
    parse_option<int>   (scf_options.diis_hist       , jscf, "diis_hist");   
    parse_option<bool>  (scf_options.force_tilesize  , jscf, "force_tilesize"); 
    parse_option<int>   (scf_options.AO_tilesize     , jscf, "tilesize");
    parse_option<double>(scf_options.alpha           , jscf, "alpha");
    parse_option<int>   (scf_options.writem          , jscf, "writem");    
    parse_option<int>   (scf_options.nnodes          , jscf, "nnodes");                                     
    parse_option<bool>  (scf_options.restart         , jscf, "restart"); 
    parse_option<bool>  (scf_options.noscf           , jscf, "noscf");     
    parse_option<bool>  (scf_options.ediis           , jscf, "ediis");
    parse_option<double>(scf_options.ediis_off       , jscf, "ediis_off");
    parse_option<bool>  (scf_options.sad             , jscf, "sad");       
    parse_option<bool>  (scf_options.debug           , jscf, "debug");
    parse_option<string>(scf_options.moldenfile      , jscf, "moldenfile"); 
    parse_option<string>(scf_options.scf_type        , jscf, "scf_type");
    parse_option<int>   (scf_options.n_lindep        , jscf, "n_lindep"); 
    parse_option<int>   (scf_options.scalapack_nb    , jscf, "scalapack_nb");
    parse_option<int>   (scf_options.scalapack_np_row, jscf, "scalapack_np_row");                                                             
    parse_option<int>   (scf_options.scalapack_np_col, jscf, "scalapack_np_col");
    
    std::string riscf_str;
    parse_option<string>(riscf_str, jscf, "riscf");
    if(riscf_str == "J")         scf_options.riscf = 1;
    else if(riscf_str == "K")    scf_options.riscf = 2;    

    //CD
    json jcd = jinput["CD"];
    parse_option<bool>  (cd_options.debug           , jcd, "debug");    
    parse_option<double>(cd_options.diagtol         , jcd, "diagtol");
    parse_option<int>   (cd_options.max_cvecs_factor, jcd, "max_cvecs");    

    //CC
    json jcc = jinput["CC"];
    parse_option<int>   (ccsd_options.ndiis         , jcc, "ndiis");  
    parse_option<int>   (ccsd_options.ccsd_maxiter  , jcc, "ccsd_maxiter");
    parse_option<double>(ccsd_options.lshift        , jcc, "lshift"); 
    parse_option<double>(ccsd_options.printtol      , jcc, "printtol"); 
    parse_option<double>(ccsd_options.threshold     , jcc, "threshold"); 
    parse_option<int>   (ccsd_options.tilesize      , jcc, "tilesize"); 
    parse_option<int>   (ccsd_options.itilesize     , jcc, "itilesize");
    parse_option<bool>  (ccsd_options.debug         , jcc, "debug");
    parse_option<bool>  (ccsd_options.readt         , jcc, "readt"); 
    parse_option<bool>  (ccsd_options.writet        , jcc, "writet");
    parse_option<int>   (ccsd_options.writet_iter   , jcc, "writet_iter");           
    parse_option<bool>  (ccsd_options.balance_tiles , jcc, "balance_tiles");
    parse_option<bool>  (ccsd_options.profile_ccsd  , jcc, "profile_ccsd");                
    parse_option<bool>  (ccsd_options.force_tilesize, jcc, "force_tilesize");     
    parse_option<string>(ccsd_options.ext_data_path , jcc, "ext_data_path");    

    json jdlpno = jcc["DLPNO"];
    parse_option<int>   (ccsd_options.max_pnos     , jdlpno, "max_pnos");
    parse_option<size_t>(ccsd_options.keep_npairs  , jdlpno, "keep_npairs");
    parse_option<bool>  (ccsd_options.localize     , jdlpno, "localize");
    parse_option<bool>  (ccsd_options.skip_dlpno   , jdlpno, "skip_dlpno");
    parse_option<string>(ccsd_options.dlpno_dfbasis, jdlpno, "df_basisset");
    parse_option<double>(ccsd_options.tcutpno      , jdlpno, "tcutpno");

    json jccsd_t = jcc["CCSD(T)"];
    parse_option<int>(ccsd_options.ngpu          , jccsd_t, "ngpu"); 
    parse_option<int>(ccsd_options.ccsdt_tilesize, jccsd_t, "ccsdt_tilesize");    

    json jeomccsd = jcc["EOMCCSD"];
    parse_option<int>   (ccsd_options.eom_nroots   , jeomccsd, "eom_nroots");   
    parse_option<int>   (ccsd_options.eom_microiter, jeomccsd, "eom_microiter");                                              
    parse_option<double>(ccsd_options.eom_threshold, jeomccsd, "eom_threshold");                 
    
    json jgfcc = jcc["GFCCSD"];
    parse_option<bool>(ccsd_options.gf_ip      , jgfcc, "gf_ip"); 
    parse_option<bool>(ccsd_options.gf_ea      , jgfcc, "gf_ea"); 
    parse_option<bool>(ccsd_options.gf_os      , jgfcc, "gf_os"); 
    parse_option<bool>(ccsd_options.gf_cs      , jgfcc, "gf_cs"); 
    parse_option<bool>(ccsd_options.gf_restart , jgfcc, "gf_restart"); 
    parse_option<bool>(ccsd_options.gf_itriples, jgfcc, "gf_itriples");

    parse_option<int>   (ccsd_options.gf_ndiis            , jgfcc, "gf_ndiis");
    parse_option<int>   (ccsd_options.gf_ngmres           , jgfcc, "gf_ngmres");
    parse_option<int>   (ccsd_options.gf_maxiter          , jgfcc, "gf_maxiter");
    parse_option<int>   (ccsd_options.gf_nprocs_poi       , jgfcc, "gf_nprocs_poi");
    parse_option<double>(ccsd_options.gf_damping_factor   , jgfcc, "gf_damping_factor");
    parse_option<double>(ccsd_options.gf_eta              , jgfcc, "gf_eta");
    parse_option<double>(ccsd_options.gf_threshold        , jgfcc, "gf_threshold");
    parse_option<double>(ccsd_options.gf_omega_min_ip     , jgfcc, "gf_omega_min_ip"); 
    parse_option<double>(ccsd_options.gf_omega_max_ip     , jgfcc, "gf_omega_max_ip");  
    parse_option<double>(ccsd_options.gf_omega_min_ip_e   , jgfcc, "gf_omega_min_ip_e");  
    parse_option<double>(ccsd_options.gf_omega_max_ip_e   , jgfcc, "gf_omega_max_ip_e");  
    parse_option<double>(ccsd_options.gf_omega_min_ea     , jgfcc, "gf_omega_min_ea");
    parse_option<double>(ccsd_options.gf_omega_max_ea     , jgfcc, "gf_omega_max_ea"); 
    parse_option<double>(ccsd_options.gf_omega_min_ea_e   , jgfcc, "gf_omega_min_ea_e");  
    parse_option<double>(ccsd_options.gf_omega_max_ea_e   , jgfcc, "gf_omega_max_ea_e");  
    parse_option<double>(ccsd_options.gf_omega_delta      , jgfcc, "gf_omega_delta");
    parse_option<double>(ccsd_options.gf_omega_delta_e    , jgfcc, "gf_omega_delta_e");
    parse_option<int>   (ccsd_options.gf_extrapolate_level, jgfcc, "gf_extrapolate_level"); 
    parse_option<int>   (ccsd_options.gf_analyze_level    , jgfcc, "gf_analyze_level");  
    parse_option<int>   (ccsd_options.gf_analyze_num_omega, jgfcc, "gf_analyze_num_omega");
    parse_option<int>   (ccsd_options.gf_p_oi_range       , jgfcc, "gf_p_oi_range"); 

    parse_option<std::vector<size_t>>(ccsd_options.gf_orbitals     , jgfcc, "gf_orbitals");
    parse_option<std::vector<double>>(ccsd_options.gf_analyze_omega, jgfcc, "gf_analyze_omega");
    
    if(ccsd_options.gf_p_oi_range!=0){
      if(ccsd_options.gf_p_oi_range != 1 && ccsd_options.gf_p_oi_range != 2)
      tamm_terminate ("gf_p_oi_range can only be one of 1 or 2");
    }

    // options.print();
    // scf_options.print();
    // ccsd_options.print();

    return std::make_tuple(options, scf_options, cd_options, ccsd_options);

}

inline std::tuple<OptionsMap, json>
   parse_input(std::istream& is) {

    const double angstrom_to_bohr =
      1.889725989; // 1 / bohr_to_angstrom; //1.889726125
    
    json jinput;
    is >> jinput;

    std::vector<string> geometry;
    parse_option<std::vector<string>>(geometry, jinput["geometry"], "coordinates", false);
    size_t natom = geometry.size();

    // rest of lines are atoms
    std::vector<Atom> atoms(natom);
    for(size_t i = 0; i < natom; i++) {
        std::string line = geometry[i];
        std::istringstream iss(line);
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

    auto [options, scf_options, cd_options, ccsd_options] = parse_json(jinput);    

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
    options_map.options.atoms = atoms;
    options_map.scf_options = scf_options;
    options_map.cd_options = cd_options;

    if(ccsd_options.eom_microiter < ccsd_options.maxiter &&
       ccsd_options.eom_microiter == options.maxiter) 
       ccsd_options.eom_microiter = ccsd_options.maxiter;

    options_map.ccsd_options = ccsd_options;

    return std::make_tuple(options_map, jinput);
}


#endif // TAMM_METHODS_INPUT_PARSER_HPP_
