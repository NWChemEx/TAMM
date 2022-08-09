#include "scf/scf_main.hpp"
#if defined(USE_UPCXX)
#include <upcxx/upcxx.hpp>
#endif

std::string filename;
using T = double;

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

    ProcGroup pg = ProcGroup::create_world_coll();
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    auto rank = ec.pg().rank();

    auto current_time = std::chrono::system_clock::now();
    auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
    auto cur_local_time = localtime(&current_time_t);

    if(rank == 0) cout << endl << "Date: " << std::put_time(cur_local_time, "%c") << endl << endl;

    // read geometry from a json file 
    json jinput;
    check_json(filename);
    auto is = std::ifstream(filename);
    
    OptionsMap options_map;
    std::tie(options_map, jinput) = parse_input(is);
    if(options_map.options.output_file_prefix.empty()) 
      options_map.options.output_file_prefix = getfilename(filename);

    // if(rank == 0) {
    //   std::ofstream res_file(getfilename(filename)+".json");
    //   res_file << std::setw(2) << jinput << std::endl;
    // }

    auto hf_t1 = std::chrono::high_resolution_clock::now();

    auto [sys_data, hf_energy, shells, shell_tile_map, C_AO, F_AO, C_beta_AO, F_beta_AO,
          AO_opt, AO_tis, scf_conv] = hartree_fock(ec, filename, options_map);

    Tensor<T>::deallocate(C_AO, F_AO);
    if(sys_data.is_ks) Tensor<T>::deallocate(vxc_tamm);
    if(sys_data.is_unrestricted) Tensor<T>::deallocate(C_beta_AO, F_beta_AO);

    if(rank == 0) {
      sys_data.output_file_prefix = options_map.options.output_file_prefix;
      sys_data.input_molecule = sys_data.output_file_prefix;
      sys_data.results["input"]["molecule"]["name"] = sys_data.output_file_prefix;
      write_json_data(sys_data,"SCF");
    }


    auto hf_t2 = std::chrono::high_resolution_clock::now();

    double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    ec.flush_and_sync();

    if(rank == 0)
    std::cout << std::endl << "Total Time taken for Hartree-Fock: " << hf_time << " secs" << std::endl;
    
    tamm::finalize();

    return 0;
}
