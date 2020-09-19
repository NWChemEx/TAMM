
#include "scf/hartree_fock_tamm.hpp"
#include "tamm/tamm.hpp"

using namespace tamm;

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

    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    
    // read geometry from a .nwx file 
    auto is = std::ifstream(filename);
    std::vector<libint2::Atom> atoms;
    OptionsMap options_map;
    std::tie(atoms, options_map) = read_input_nwx(is);

    if(options_map.options.output_file_prefix.empty()) 
      options_map.options.output_file_prefix = getfilename(filename);
    
    auto hf_t1 = std::chrono::high_resolution_clock::now();

    hartree_fock(ec, filename, atoms, options_map);

    auto hf_t2 = std::chrono::high_resolution_clock::now();

    double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    ec.flush_and_sync();

    if(GA_Nodeid() == 0)
    std::cout << std::endl << "Total Time taken for Hartree-Fock: " << hf_time << " secs" << std::endl;
    
    tamm::finalize();

    return 0;
}
