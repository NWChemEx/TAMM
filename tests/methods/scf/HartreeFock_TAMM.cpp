#define CATCH_CONFIG_RUNNER

#include "scf/hartree_fock_tamm.hpp"
#include "catch/catch.hpp"
#include "tamm/tamm.hpp"

using namespace tamm;


std::string filename;

TEST_CASE("HartreeFock testcase") {
    // Matrix C;
    // Matrix F;

    ProcGroup pg = ProcGroup::create_coll(GA_MPI_Comm());
    auto mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    RuntimeEngine re;
    ExecutionContext ec{pg, &distribution, mgr, &re};
    
    auto hf_t1 = std::chrono::high_resolution_clock::now();
    // std::tie(ov_alpha, nao, hf_energy, shells) = hartree_fock(filename, C, F);

    // read geometry from a .nwx file 
    auto is = std::ifstream(filename);
    std::vector<libint2::Atom> atoms;
    OptionsMap options_map;
    std::tie(atoms, options_map) = read_input_nwx(is);

    CHECK_NOTHROW(hartree_fock(ec, filename, atoms, options_map));
    auto hf_t2 = std::chrono::high_resolution_clock::now();

    double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();

    ec.flush_and_sync();
    MemoryManagerGA::destroy_coll(mgr);
    // delete ec;

    if(GA_Nodeid() == 0)
    std::cout << "\nTotal Time taken for Hartree-Fock: " << hf_time << " secs\n";
}

int main( int argc, char* argv[] )
{
    if(argc<2){
        std::cout << "Please provide an input file!\n";
        return 1;
    }

    filename = std::string(argv[1]);

    std::ifstream testinput(filename); 
    if(!testinput){
        std::cout << "Input file provided [" << filename << "] does not exist!\n";
        return 1;
    }

    MPI_Init(&argc,&argv);
    GA_Initialize();
    MA_init(MT_DBL, 8000000, 20000000);
    
    int mpi_rank;
    MPI_Comm_rank(GA_MPI_Comm(), &mpi_rank);
    #ifdef USE_TALSH
    TALSH talsh_instance;
    talsh_instance.initialize(mpi_rank);
    #endif

    int res = Catch::Session().run();

    #ifdef USE_TALSH
    talsh_instance.shutdown();
    #endif    
    GA_Terminate();
    MPI_Finalize();

    return res;
}
