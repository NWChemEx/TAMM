#include "macdecls.h"
#include "mpi.h"
#include <chrono>
#include <tamm/tamm.hpp>

using namespace tamm;

template<typename T>
void test_2_dim_mult_op(Scheduler& sch, size_t N, Tile tilesize) {
    TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};

    auto [i, j, k] = tis1.labels<3>("all");

    Tensor<T> A{i, k};
    Tensor<T> B{k, j};
    Tensor<T> C{i, j};

    sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

    const auto timer_start = std::chrono::high_resolution_clock::now();

    sch(C(j, i) += A(i, k) * B(k, j)).execute();

    const auto timer_end = std::chrono::high_resolution_clock::now();

    auto mult_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                       (timer_end - timer_start))
                       .count();

    if(sch.ec().pg().rank() == 0)
        std::cout << "2-D Tensor contraction with " << N
                  << " indices tiled with " << tilesize << " : " << mult_time
                  << std::endl;
}

template<typename T>
void test_3_dim_mult_op(Scheduler& sch, size_t N, Tile tilesize) {
    TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};

    auto [i, j, k, l, m] = tis1.labels<5>("all");

    Tensor<T> A{i, j, l};
    Tensor<T> B{l, m, k};
    Tensor<T> C{i, j, k};

    sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

    const auto timer_start = std::chrono::high_resolution_clock::now();

    sch(C(j, i, k) += A(i, j, l) * B(l, m, k)).execute();

    const auto timer_end = std::chrono::high_resolution_clock::now();

    auto mult_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                       (timer_end - timer_start))
                       .count();

    if(sch.ec().pg().rank() == 0)
        std::cout << "3-D Tensor contraction with " << N
                  << " indices tiled with " << tilesize << " : " << mult_time
                  << std::endl;
}

template<typename T>
void test_4_dim_mult_op(Scheduler& sch, size_t N, Tile tilesize) {
    TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};

    auto [i, j, k, l, m, o] = tis1.labels<6>("all");

    Tensor<T> A{i, j, m, o};
    Tensor<T> B{m, o, k, l};
    Tensor<T> C{i, j, k, l};

    sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

    const auto timer_start = std::chrono::high_resolution_clock::now();

    sch(C(j, i, k, l) += A(i, j, m, o) * B(m, o, k, l)).execute();

    const auto timer_end = std::chrono::high_resolution_clock::now();

    auto mult_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                       (timer_end - timer_start))
                       .count();

    if(sch.ec().pg().rank() == 0)
        std::cout << "4-D Tensor contraction with " << N
                  << " indices tiled with " << tilesize << " : " << mult_time
                  << std::endl;
}

int main(int argc, char* argv[]) {
    if(argc < 3) {
        std::cout << "Please provide an index space size and tile size!\n";
        return 0;
    }

    size_t is_size = atoi(argv[1]);
    Tile tile_size = atoi(argv[2]);

    if(is_size < tile_size) {
        std::cout << "Tile size should be less then index space size"
                  << std::endl;
        return 1;
    }

    MPI_Init(&argc, &argv);
    GA_Initialize();
    MA_init(MT_DBL, 8000000, 20000000);

    int mpi_rank;
    MPI_Comm_rank(GA_MPI_Comm(), &mpi_rank);
    #ifdef USE_TALSH
    TALSH talsh_instance;
    talsh_instance.initialize(mpi_rank);
    #endif

    ProcGroup pg{GA_MPI_Comm()};
    MemoryManagerGA* mgr = MemoryManagerGA::create_coll(pg);
    Distribution_NW distribution;
    RuntimeEngine re;
    ExecutionContext ec{pg, &distribution, mgr, &re};

    Scheduler sch{ec};

    test_2_dim_mult_op<double>(sch, is_size, tile_size);
    test_3_dim_mult_op<double>(sch, is_size, tile_size);
    test_4_dim_mult_op<double>(sch, is_size, tile_size);

    #ifdef USE_TALSH
    talsh_instance.shutdown();
    #endif

    GA_Terminate();
    MPI_Finalize();

    return 0;
}
