#define CATCH_CONFIG_RUNNER

#include "HF/hartree_fock.hpp"
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
    // Tensor4D V2;
    // TAMM_SIZE ov_alpha{0};
    // TAMM_SIZE freeze_core    = 0;
    // TAMM_SIZE freeze_virtual = 0;

    // double hf_energy{0.0};
    // libint2::BasisSet shells;
    // TAMM_SIZE nao{0};

    auto hf_t1 = std::chrono::high_resolution_clock::now();
    // std::tie(ov_alpha, nao, hf_energy, shells) = hartree_fock(filename, C, F);
    CHECK_NOTHROW(hartree_fock(filename, C, F));
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
