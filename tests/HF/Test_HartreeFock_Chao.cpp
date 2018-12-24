#define CATCH_CONFIG_RUNNER

#include "HF/hartree_fock_chao.hpp"
#include "catch/catch.hpp"
#include "tamm/tamm.hpp"
#include "macdecls.h"
#include "ga-mpi.h"


using namespace tamm;

std::string filename;

TEST_CASE("HartreeFock testcase") {
    using T = double;

    Matrix C;
    Matrix F;

    auto hf_t1 = std::chrono::high_resolution_clock::now();
    // read geometry from a .nwx file 
    auto is = std::ifstream(filename);
    std::vector<libint2::Atom> atoms;
    std::unordered_map<std::string, Options> options_map;
    std::tie(atoms, options_map) = read_input_nwx(is);

    CHECK_NOTHROW(hartree_fock(ec, filename, atoms, options_map));
    
    auto hf_t2 = std::chrono::high_resolution_clock::now();

    double hf_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((hf_t2 - hf_t1)).count();
    std::cout << "\nTime taken for Hartree-Fock: " << hf_time << " secs\n";
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
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int res = Catch::Session().run();
    
    GA_Terminate();
    MPI_Finalize();

    return res;
}
