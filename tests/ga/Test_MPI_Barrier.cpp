
#include "macdecls.h"
#include "mpi.h"
#include "armci.h"
#include "ga.h"
#include "ga-mpi.h"

#include <iostream>

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    double tbeg = MPI_Wtime();
    auto ierr = MPI_Barrier(MPI_COMM_WORLD);
    double t_barrier_mpi = MPI_Wtime()-tbeg;

    int nproc;
    int wrank;
    double gbarrier;
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Reduce(&t_barrier_mpi,&gbarrier,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    if(wrank==0) std::cout << "Time (s) for MPI Barrier on MPI_COMM_WORLD with " << nproc << " ranks: " << gbarrier/((double)nproc) << std::endl;

    nproc = 0; gbarrier = 0;
    
    GA_Initialize();
    MA_init(MT_DBL, 8000000, 20000000);

    tbeg = MPI_Wtime();
    MPI_Barrier(GA_MPI_Comm());
    double t_barrier_ga = MPI_Wtime()-tbeg;

    int my_rank;
    MPI_Comm_rank(GA_MPI_Comm(),&my_rank);
    ierr = MPI_Comm_size(GA_MPI_Comm(),&nproc);

    MPI_Reduce(&t_barrier_ga,&gbarrier,1,MPI_DOUBLE,MPI_SUM,0,GA_MPI_Comm());
    if(my_rank==0) std::cout << "Time (s) for MPI Barrier on GA_MPI_Comm with " << nproc << " ranks: " << gbarrier/((double)nproc) << std::endl;

    GA_Terminate();
    MPI_Finalize();

    return 0;
}
