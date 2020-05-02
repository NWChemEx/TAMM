
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
    int mpi_rank;
    MPI_Comm_rank(GA_MPI_Comm(), &mpi_rank);
    #ifdef USE_TALSH
    TALSH talsh_instance;
    talsh_instance.initialize(mpi_rank);
    #endif

    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    RuntimeEngine re;
    ExecutionContext ec{pg, &distribution, mgr, &re};
    
    auto hf_t1 = std::chrono::high_resolution_clock::now();

    // read geometry from a .nwx file 
    auto is = std::ifstream(filename);
    std::vector<libint2::Atom> atoms;
    OptionsMap options_map;
    std::tie(atoms, options_map) = read_input_nwx(is);

    hartree_fock(ec, filename, atoms, options_map);
    auto hf_t2 = std::chrono::high_resolution_clock::now();

    double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    ec.flush_and_sync();
    MemoryManagerGA::destroy_coll(mgr);
    // delete ec;

    if(GA_Nodeid() == 0)
    std::cout << std::endl << "Total Time taken for Hartree-Fock: " << hf_time << " secs" << std::endl;

    #ifdef USE_TALSH
    talsh_instance.shutdown();
    #endif    
    
    tamm::finalize();

    return 0;
}