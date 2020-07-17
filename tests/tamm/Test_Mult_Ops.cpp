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

    // sch.allocate(A, B, C).execute();
    // sch(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();
    sch.allocate(A).execute();

    std::vector<Index> itr;
    const Distribution_Dense& dd =
        static_cast<const Distribution_Dense&>(A.distribution());
    // dd.iterate(
    //     {0, 0}, [&]() { std::cout << "itr=" << itr << "\n"; }, itr);
    const auto timer_start = std::chrono::high_resolution_clock::now();
    sch(A() = 21.0).execute();

    // sch(C(j, i) += A(i, k) * B(k, j)).execute();

    const auto timer_end = std::chrono::high_resolution_clock::now();

    auto mult_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                         (timer_end - timer_start))
                         .count();

    if (sch.ec().pg().rank() == 0)
      std::cout << "2-D Tensor contraction with " << N << " indices tiled with "
                << tilesize << " : " << mult_time << std::endl;
}

template <typename T>
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

  if (sch.ec().pg().rank() == 0)
    std::cout << "3-D Tensor contraction with " << N << " indices tiled with "
              << tilesize << " : " << mult_time << std::endl;
}

template <typename T>
void test_4_dim_mult_op(Scheduler& sch, size_t N, Tile tilesize) {
  TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};

  auto [i, j, k, l, m, o] = tis1.labels<6>("all");

  Tensor<T> A{i, j, m, o};
  Tensor<T> B{m, o, k, l};
  Tensor<T> C{i, j, k, l};

  sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

  const auto timer_start = std::chrono::high_resolution_clock::now();

  sch.exact_copy(A(i, j, m, o), B(m, o, k, l))
  (C(j, i, k, l) += A(i, j, m, o) * B(m, o, k, l)).execute();

  const auto timer_end = std::chrono::high_resolution_clock::now();

  auto mult_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                       (timer_end - timer_start))
                       .count();

  if (sch.ec().pg().rank() == 0)
    std::cout << "4-D Tensor contraction with " << N << " indices tiled with "
              << tilesize << " : " << mult_time << std::endl;
}

template <typename T>
void test_4_dim_mult_op_last_unit(Scheduler& sch, size_t N, Tile tilesize) {
  TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};
  size_t size = N / 10 > 0 ? N / 10 : 1;
  TiledIndexSpace tis2{IndexSpace{range(size)}};

  auto [i, j, k, l] = tis1.labels<4>("all");
  auto [m, o] = tis2.labels<2>("all");

  Tensor<T> A{m, o};
  Tensor<T> B{i, j, k, m};
  Tensor<T> C{i, j, k, m};

  sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

  const auto timer_start = std::chrono::high_resolution_clock::now();

  sch(C(j, i, k, o) += A(m, o) * B(i, j, k, m)).execute();

  const auto timer_end = std::chrono::high_resolution_clock::now();

  auto mult_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                       (timer_end - timer_start))
                       .count();

  if (sch.ec().pg().rank() == 0)
    std::cout << "4-D Tensor contraction with 2-D unit tiled matrix " << N
              << " indices tiled with " << tilesize
              << " last index unit tiled : " << mult_time << std::endl;
}

template <typename T>
void test_4_dim_mult_op_first_unit(Scheduler& sch, size_t N, Tile tilesize) {
  TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};
  size_t size = N / 10 > 0 ? N / 10 : 1;
  TiledIndexSpace tis2{IndexSpace{range(size)}};
  TiledIndexSpace tis3{IndexSpace{range(tilesize)}, tilesize};

  auto [i, j, k, l, m, o] = tis1.labels<6>("all");
  auto [t1, t2] = tis3.labels<2>("all");
  // auto [m, o] = tis2.labels<2>("all");

  Tensor<T> A{m, t1};
  Tensor<T> B{t1, m};
  Tensor<T> C{m, i};

  sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

  const auto timer_start = std::chrono::high_resolution_clock::now();

  sch(C(m, i) += A(m, t1) * B(t1, i)).execute();

  const auto timer_end = std::chrono::high_resolution_clock::now();

  auto mult_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                       (timer_end - timer_start))
                       .count();

  if (sch.ec().pg().rank() == 0)
    std::cout << "4-D Tensor contraction with 2-D unit tiled matrix " << N
              << " indices tiled with " << tilesize
              << " first index unit tiled : " << mult_time << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Please provide an index space size and tile size!\n";
    return 0;
  }

  size_t is_size = atoi(argv[1]);
  Tile tile_size = atoi(argv[2]);

  if (is_size < tile_size) {
    std::cout << "Tile size should be less then index space size" << std::endl;
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
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  Scheduler sch{ec};

  int O = 30; //292/2;
  int V = 103; //1183/2;
  int tilesize = tile_size;
  std::cerr<<"tilesize="<<tilesize<<"\n";
  std::vector<Tile> tilesizes_o, tilesizes_v;
  int64_t ntiles_o = static_cast<int64_t>(std::ceil(1.0 * O / tilesize));
  int64_t ntiles_v = static_cast<int64_t>(std::ceil(1.0 * V / tilesize));
  Size ctr = 0;
  for(int i=0; i<ntiles_o; i++) {
    tilesizes_o.push_back(O / ntiles_o + (i<(O % ntiles_o)));
    ctr += tilesizes_o.back();
  }
  EXPECTS(ctr == O);
  ctr = 0;
  for (int i = 0; i < ntiles_v; i++) {
    tilesizes_v.push_back(V / ntiles_v + (i < (V % ntiles_v)));
    ctr += tilesizes_v.back();
  }
  EXPECTS(ctr == V);
  TiledIndexSpace tis_o{IndexSpace{range(O)}, static_cast<tamm::Tile>(tilesize)};
  TiledIndexSpace tis_v{IndexSpace{range(V)}, static_cast<tamm::Tile>(tilesize)};
  TiledIndexSpace tis_o_balanced{IndexSpace{range(O)}, tilesizes_o};
  TiledIndexSpace tis_v_balanced{IndexSpace{range(V)}, tilesizes_v};

  auto [i, j, k, l] = tis_o.labels<4>("all");
  auto [a, b, c, d] = tis_v.labels<4>("all");

  auto [i1, j1, k1, l1] = tis_o_balanced.labels<4>("all");
  auto [a1, b1, c1, d1] = tis_v_balanced.labels<4>("all");

  {
    using T = double;
    Tensor<std::complex<T>> A1{i,j,a,b,k};
    std::cerr << __FUNCTION__ << " " << __LINE__ << "\n";    
    sch.allocate(A1).execute();
    std::cerr << __FUNCTION__ << " " << __LINE__ << "\n";
    sch.deallocate(A1).execute();
    std::cout << "5d test works \n";

    Tensor<T> A{i, j, a, b};

    sch.allocate(A).execute();

    Size min_block_size = 1000000000l, max_block_size = 0;
    for (const auto& blockid : A.loop_nest()) {
      if (A.is_non_zero(blockid)) {
        Size sz = A.block_size(blockid);
        min_block_size = std::min(min_block_size, sz);
        max_block_size = std::max(max_block_size, sz);
      }
    }
    std::cout << "min block size = " << min_block_size << "\n"
              << "max block size = " << max_block_size << "\n";
  }
  {
    using T = double;
    Tensor<T> A{i1, j1, a1, b1};

    sch.allocate(A).execute();

    Size min_block_size = 1000000000l, max_block_size = 0;
    for (const auto& blockid : A.loop_nest()) {
      if (A.is_non_zero(blockid)) {
        Size sz = A.block_size(blockid);
        min_block_size = std::min(min_block_size, sz);
        max_block_size = std::max(max_block_size, sz);
      }
    }
    std::cout << "min block size = " << min_block_size << "\n"
              << "max block size = " << max_block_size << "\n";
  }

//#if 0
  test_2_dim_mult_op<double>(sch, is_size, tile_size);
  test_3_dim_mult_op<double>(sch, is_size, tile_size);
  test_4_dim_mult_op<double>(sch, is_size, tile_size);
#if 0
  test_4_dim_mult_op_last_unit<double>(sch, is_size, tile_size);
  test_4_dim_mult_op_first_unit<double>(sch, is_size, tile_size);
#endif
  std::cout << "multOpDgemmTime=" << multOpDgemmTime << "\n";

#ifdef USE_TALSH
  talsh_instance.shutdown();
#endif

  tamm::finalize();

  return 0;
}
