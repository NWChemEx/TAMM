
#include "tamm/tamm.hpp"
#include "tamm/utils.hpp"
#include "macdecls.h"
#include "ga-mpi.h"


using namespace tamm;

std::string ccsd_test( int argc, char* argv[] )
{

    if(argc<2){
        std::cout << "Please provide an input file!\n";
        exit(0);
    }

    auto filename = std::string(argv[1]);
    std::ifstream testinput(filename); 
    if(!testinput){
        std::cout << "Input file provided [" << filename << "] does not exist!\n";
        exit(0);
    }

    return filename;
}

