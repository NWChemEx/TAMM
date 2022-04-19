#include "ccsd_util.hpp"

#include <filesystem>
namespace fs = std::filesystem;

void cd_driver();
std::string filename;

int main( int argc, char* argv[] )
{
    if(argc<2){
        std::cout << "Please provide an input file!" << std::endl;
        return 1;
    }

    filename = std::string(argv[1]);
    std::ifstream testinput(filename); 
    if(!testinput){
        std::cout << "Input file provided [" << filename << "] does not exist!" << std::endl;
        return 1;
    }

    tamm::initialize(argc, argv);

    cd_driver();

    tamm::finalize();

    return 0;
}

void cd_driver() {

    // std::cout << "Input file provided = " << filename << std::endl;

    using T = double;

    ProcGroup pg = ProcGroup::create_world_coll();
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    auto rank = ec.pg().rank();

    auto [sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, C_beta_AO, F_beta_AO, AO_opt, AO_tis,scf_conv]  
                    = hartree_fock_driver<T>(ec,filename);

    CCSDOptions ccsd_options = sys_data.options_map.ccsd_options;

    if(rank == 0) ccsd_options.print();

    if(rank==0) cout << endl << "#occupied, #virtual = " << sys_data.nocc << ", " << sys_data.nvir << endl;
    
    auto [MO,total_orbitals] = setupMOIS(sys_data);

    std::string out_fp = sys_data.output_file_prefix+"."+ccsd_options.basis;
    std::string files_dir = out_fp+"_files/"+sys_data.options_map.scf_options.scf_type;
    std::string files_prefix = /*out_fp;*/ files_dir+"/"+out_fp;
    std::string f1file = files_prefix+".f1_mo";
    std::string v2file = files_prefix+".cholv2";
    std::string cholfile = files_prefix+".cholcount";
    
    bool cd_restart = fs::exists(f1file) && fs::exists(v2file) && fs::exists(cholfile);

    //deallocates F_AO, C_AO
    auto [cholVpr,d_f1,lcao,chol_count, max_cvecs, CI] = cd_svd_ga_driver<T>
                        (sys_data, ec, MO, AO_opt, C_AO, F_AO, C_beta_AO, F_beta_AO, shells, shell_tile_map,
                                cd_restart, cholfile);
    free_tensors(lcao);

    if(!cd_restart && ccsd_options.writet) {
        if(!fs::exists(files_dir)) fs::create_directories(files_dir);

        write_to_disk(d_f1,f1file);
        write_to_disk(cholVpr,v2file);

        if(rank==0){
          std::ofstream out(cholfile, std::ios::out);
          if(!out) cerr << "Error opening file " << cholfile << endl;
          out << chol_count << std::endl;
          out.close();
        }        
    }

    free_tensors(d_f1,cholVpr);

    ec.flush_and_sync();
    // delete ec;

}
