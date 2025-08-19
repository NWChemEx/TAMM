#include <chrono>
#include <tamm/tamm.hpp>
#include <tamm/tamm_git.hpp>

using namespace tamm;

template<typename T>
void test_2_dim_mult_op(Scheduler& sch, size_t N, Tile tilesize, ExecutionHW ex_hw, bool profile) {
  TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};

  auto [i, j, k] = tis1.labels<3>("all");

  Tensor<T> A{i, k};
  Tensor<T> B{k, j};
  Tensor<T> C{i, j};

  sch.allocate(A, B, C).execute();
  sch(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

  const auto timer_start = std::chrono::high_resolution_clock::now();

  sch(C(j, i) += A(i, k) * B(k, j)).execute(ex_hw, profile);

  const auto timer_end = std::chrono::high_resolution_clock::now();

  auto mult_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

  if(sch.ec().print())
    std::cout << "2-D Tensor contraction with " << N << " indices tiled with " << tilesize << " : "
              << mult_time << std::endl;

  sch.deallocate(A, B, C).execute();

  // 2D dense case
  // ExecutionContext ec{pg, DistributionKind::dense, MemoryManagerKind::ga};
  // using Matrix   = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  // TiledIndexSpace AO{IndexSpace{range(N)}, 50};
  // TiledIndexSpace AO{IndexSpace{range(N)}, {2,2,1}};
  // auto [mu,nu,ku] = AO.labels<3>("all");
  // Tensor<T> X1{AO,AO};
  // Tensor<T> X2{AO,AO};
  // Tensor<T> X3{AO,AO};
  // X1.set_dense();
  // X2.set_dense();
  // X3.set_dense();
  // Scheduler{ec}.allocate(X1,X2,X3)
  // (X2()=3.2)
  // (X3()=2.4)
  // (X1(mu,nu) = X2(mu,ku) * X3(ku,nu)).execute();

  // if(pg.rank()==0) {
  //   Matrix X1_e(N,N);
  //   std::cout << "----------------" << std::endl;
  //   X1_e.setRandom();
  //   std::cout << X1_e << std::endl;
  //   eigen_to_tamm_tensor<double>(X1,X1_e);
  // }
  // Tensor<T> X1_b = tensor_block(X1, {0,0}, {2,2});
  // Tensor<T> X1_bp = permute_tensor(X1_b,{1,0});

  // if(pg.rank()==0) {
  //   std::cout << "----------------" << std::endl;
  //   Matrix X_eig = tamm_to_eigen_matrix<double>(X1_b);
  //   std::cout << X_eig << std::endl;
  //   X_eig = tamm_to_eigen_matrix<double>(X1_bp);
  //   std::cout << "----------------" << std::endl;
  //   std::cout << X_eig << std::endl;
  // }

  // sch.deallocate(X1,X2,X3,X1_b,X1_bp).execute();
  // X1 = {AO,AO,AO};
  // X2 = {AO,AO,AO};
  // X3 = {AO,AO,AO};
  // Tensor<T>::set_dense(X1,X2,X3);
  // sch.allocate(X1,X2,X3).execute();
  // sch(X2()=3.2)
  // (X3()=2.3)
  // (X1(mu,nu,ku) = X2(mu,nu,lu)* X3(lu,nu,ku)).execute();

  // if(pg.rank()==0) {
  //   std::cout << "----------------" << std::endl;
  //   Tensor3D X1_e(N,N,N);
  //   X1_e.setRandom();
  //   eigen_to_tamm_tensor<double>(X1,X1_e);
  //   std::cout << X1_e << std::endl;
  // }

  // X1_b = tensor_block(X1, {0,0,0}, {2,1,2});
  // if(pg.rank() == 0) std::cout << "X1_b is BC: " << X1_b.is_block_cyclic() << std::endl;
  // Tensor<T> X1_bp3 = permute_tensor(X1_b,{2,0,1});
  // if(pg.rank()==0) {
  //   std::cout << "----------------" << std::endl;
  //   Tensor3D X_eig = tamm_to_eigen_tensor<double,3>(X1_b);
  //   std::cout << X_eig << std::endl;
  //   std::cout << "----------------" << std::endl;
  //   X_eig = tamm_to_eigen_tensor<double,3>(X1_bp3);
  //   std::cout << X_eig.dimensions() << std::endl;
  //   std::cout << X_eig << std::endl;
  // }

  // std::random_device rd;
  // std::mt19937                     e2(rd());
  // std::uniform_real_distribution<> dist(0, 25);
  // double* fptr = F_BC.access_local_buf();
  // for(int i=0;i<F_BC.local_buf_size();i++)
  //  fptr[i] = dist(e2);

  // std::cout << "rank " << scalapack_info.pg.rank().value() << std::endl;
  // for(int i=0;i<F_BC.local_buf_size();i++) {
  //  if(i%11==0) std::cout << "\n";
  //  std::cout << fptr[i] << "\t";
  // }
  // std::cout << "\n";

  // // print_tensor(F_BC);
  // // if(scalapack_info.pg.rank() == 0) print_tensor(F_BC);
  // Tensor<T> F1 = from_block_cyclic_tensor(F_BC);
  // if(scalapack_info.pg.rank() == 0) std::cout << "f1 pg = " << F1.proc_grid()[0].value() << ","
  // << F1.proc_grid()[1].value() << "\n"; if(scalapack_info.pg.rank() == 0) print_tensor(F1);
  // // eigen_to_tamm_tensor<double>(F1,X1_e);

  // Tensor<T> X1_bp3 = tensor_block(F1, {8,0}, {11,11}, {1,0});
  // if(scalapack_info.pg.rank() == 0) {
  //   std::cout << X1_e << "\n";
  //   std::cout << std::string(16,'-') << "\n";
  //   X1_e = tamm_to_eigen_matrix<double>(X1_bp3);
  //   std::cout << X1_e << "\n";
  // }
}


fastcc::ListTensor<double> make_sparse_tensor(std::vector<int>& shape, float density=0.1) {
  fastcc::ListTensor<double> tensor(shape.size());
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  uint64_t dense_span = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  int num_nnzs = std::ceil(dense_span * density );
  std::unordered_set<int> nnz_indices;
  for(int nnz_idx = 0; nnz_idx < num_nnzs; nnz_idx++) {
    std::vector<int> coords;
    int dense_pos = nnz_idx * dense_span / num_nnzs;
    while(nnz_indices.find(dense_pos) != nnz_indices.end()){
        dense_pos = (dense_pos + 1) % dense_span;
    }
    nnz_indices.insert(dense_pos);
    for(int i = 0; i < shape.size(); i++) {
      coords.push_back(dense_pos % shape[i]);
      dense_pos /= shape[i];
    }
    tensor.push_nnz(dis(gen), fastcc::CompactCordinate(tamm::fastcc::CoOrdinate(coords)));
  }
  tensor.set_shape(shape);
  std::cout<<"created tensor with "<<num_nnzs<<" nnzs"<<std::endl;
  
  return tensor;
}

template<typename T>
void test_3_dim_mult_op(Scheduler& sch, size_t N, Tile tilesize, ExecutionHW ex_hw, bool profile) {
  TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};
  TiledIndexSpace tis2{IndexSpace{range(N/5)}, tilesize};
  TiledIndexSpace tis3{IndexSpace{range(N/10)}, tilesize};

  auto [i, k] = tis1.labels<2>("all");
  auto [j, l] = tis2.labels<2>("all");
  auto [m]    = tis3.labels<1>("all");

  Tensor<T> A{i, j, l};
  Tensor<T> B{l, m, k};
  Tensor<T> C{i, m, j, k};
  Tensor<T> D{m, k, l};

  sch.allocate(A, B, C, D)(A() = 21.0)(B() = 2.0)(C() = 0.0)(D() = 0.0).execute();
  fastcc::init_heaps(1);
  auto shape = std::vector<int>{static_cast<int>(N), static_cast<int>(N/5), static_cast<int>(N/5)};
  auto a_list_tensor = make_sparse_tensor(shape);
  //auto a_fastcc_tensor = a_list_tensor.to_tensor();
  //a_fastcc_tensor._infer_dimensionality();
  //a_fastcc_tensor.set_shape(shape);
  A.set_listtensor(a_list_tensor);
  //A.set_fastcctensor(a_fastcc_tensor);
  shape = std::vector<int>{static_cast<int>(N/5), static_cast<int>(N/10), static_cast<int>(N)};
  auto b_list_tensor = make_sparse_tensor(shape);
  auto b_fastcc_tensor = b_list_tensor.to_tensor();
  b_fastcc_tensor._infer_dimensionality();
  b_fastcc_tensor.set_shape(shape);
  //B.set_listtensor(b_list_tensor);
  B.set_fastcctensor(b_fastcc_tensor);

  auto timer_start = std::chrono::high_resolution_clock::now();

  sch(C(i, m, j, k) += A(i, j, l) * B(l, m, k)).execute(ExecutionHW::CPU_SPARSE, profile);
  //fastcc will write this as C(j, i, k, m) no matter what
  sch(D(m, k, l) += C(i, m, j, k) * A(i, j, l)).execute(ExecutionHW::CPU_SPARSE, profile);

  auto timer_end = std::chrono::high_resolution_clock::now();

  auto mult_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

  if(sch.ec().print())
    std::cout << "3-D sparse Tensor contraction with " << N << " indices tiled with " << tilesize << " : "
              << mult_time << std::endl;


  timer_start = std::chrono::high_resolution_clock::now();

  //sch(C(i, j, m, k) += A(i, j, l) * B(l, m, k)).execute(ExecutionHW::CPU, profile);

  timer_end = std::chrono::high_resolution_clock::now();

  mult_time = std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

  if(sch.ec().print())
    std::cout << "3-D dense Tensor contraction with " << N << " indices tiled with " << tilesize << " : "
              << mult_time << std::endl;

  sch.deallocate(A, B, C).execute();
  fastcc::destroy_heaps(1);
}

template<typename T>
void norm_check(Tensor<T> tensor, bool ci_check) {
  if(!ci_check) return;
  T      tnorm = tamm::norm(tensor);
  double tval  = 0;
  if constexpr(tamm::internal::is_complex_v<T>) tval = tnorm.real();
  else tval = tnorm;
  const bool mop_pass = (std::fabs(tval - 2.625e8) <= 1e-9);
  if(!mop_pass) {
    if(tensor.execution_context()->print())
      std::cout << "norm value: " << tval << ", expected: 2.625e8" << std::endl;
    EXPECTS(mop_pass);
  }
}

template<typename T>
void test_4_dim_mult_op(Scheduler& sch, size_t N, Tile tilesize, ExecutionHW ex_hw, bool profile) {
  TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};

  using CT = T;

  auto [i, j, k, l, m, o] = tis1.labels<6>("all");

  {
    Tensor<T> A{i, j, m, o};
    Tensor<T> B{m, o, k, l};
    Tensor<T> C{i, j, k, l};

    sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

    const auto timer_start = std::chrono::high_resolution_clock::now();

    sch //.exact_copy(A(i, j, m, o), B(m, o, k, l))
      (C(j, i, k, l) += A(i, j, m, o) * B(m, o, k, l))
        .execute(ex_hw, profile);

    const auto timer_end = std::chrono::high_resolution_clock::now();

    auto mult_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

    if(sch.ec().print())
      std::cout << "4-D Tensor contraction (R=RxR) with " << N << " indices tiled with " << tilesize
                << " : " << mult_time << std::endl;
    norm_check(C, N == 50);
    sch.deallocate(A, B, C).execute();
  }

  {
    Tensor<T>  A{i, j, m, o};
    Tensor<T>  B{m, o, k, l};
    Tensor<CT> C{i, j, k, l};

    sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

    const auto timer_start = std::chrono::high_resolution_clock::now();

    sch //.exact_copy(A(i, j, m, o), B(m, o, k, l))
      (C(j, i, k, l) += A(i, j, m, o) * B(m, o, k, l))
        .execute(ex_hw, profile);

    const auto timer_end = std::chrono::high_resolution_clock::now();

    auto mult_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

    if(sch.ec().print())
      std::cout << "4-D Tensor contraction (C=RxR) with " << N << " indices tiled with " << tilesize
                << " : " << mult_time << std::endl;

    norm_check(C, N == 50);
    sch.deallocate(A, B, C).execute();
  }

  {
    Tensor<T>  A{i, j, m, o};
    Tensor<CT> B{m, o, k, l};
    Tensor<CT> C{i, j, k, l};

    sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

    const auto timer_start = std::chrono::high_resolution_clock::now();

    sch //.exact_copy(A(i, j, m, o), B(m, o, k, l))
      (C(j, i, k, l) += A(i, j, m, o) * B(m, o, k, l))
        .execute(ex_hw, profile);

    const auto timer_end = std::chrono::high_resolution_clock::now();

    auto mult_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

    if(sch.ec().print())
      std::cout << "4-D Tensor contraction (C=RxC) with " << N << " indices tiled with " << tilesize
                << " : " << mult_time << std::endl;

    norm_check(C, N == 50);
    sch.deallocate(A, B, C).execute();
  }

  {
    Tensor<CT> A{i, j, m, o};
    Tensor<T>  B{m, o, k, l};
    Tensor<CT> C{i, j, k, l};

    sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

    const auto timer_start = std::chrono::high_resolution_clock::now();

    sch //.exact_copy(A(i, j, m, o), B(m, o, k, l))
      (C(j, i, k, l) += A(i, j, m, o) * B(m, o, k, l))
        .execute(ex_hw, profile);

    const auto timer_end = std::chrono::high_resolution_clock::now();

    auto mult_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

    if(sch.ec().print())
      std::cout << "4-D Tensor contraction (C=CxR) with " << N << " indices tiled with " << tilesize
                << " : " << mult_time << std::endl;

    norm_check(C, N == 50);
    sch.deallocate(A, B, C).execute();
  }

  {
    Tensor<CT> A{i, j, m, o};
    Tensor<T>  B{m, o, k, l};
    Tensor<T>  C{i, j, k, l};

    sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

    const auto timer_start = std::chrono::high_resolution_clock::now();

    sch //.exact_copy(A(i, j, m, o), B(m, o, k, l))
      (C(j, i, k, l) += A(i, j, m, o) * B(m, o, k, l))
        .execute(ex_hw, profile);

    const auto timer_end = std::chrono::high_resolution_clock::now();

    auto mult_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

    if(sch.ec().print())
      std::cout << "4-D Tensor contraction (R=CxR) with " << N << " indices tiled with " << tilesize
                << " : " << mult_time << std::endl;

    norm_check(C, N == 50);
    sch.deallocate(A, B, C).execute();
  }

  {
    Tensor<T>  A{i, j, m, o};
    Tensor<CT> B{m, o, k, l};
    Tensor<T>  C{i, j, k, l};

    sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

    const auto timer_start = std::chrono::high_resolution_clock::now();

    sch //.exact_copy(A(i, j, m, o), B(m, o, k, l))
      (C(j, i, k, l) += A(i, j, m, o) * B(m, o, k, l))
        .execute(ex_hw, profile);

    const auto timer_end = std::chrono::high_resolution_clock::now();

    auto mult_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

    if(sch.ec().print())
      std::cout << "4-D Tensor contraction (R=RxC) with " << N << " indices tiled with " << tilesize
                << " : " << mult_time << std::endl;

    norm_check(C, N == 50);
    sch.deallocate(A, B, C).execute();
  }

  {
    Tensor<CT> A{i, j, m, o};
    Tensor<CT> B{m, o, k, l};
    Tensor<CT> C{i, j, k, l};

    sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

    const auto timer_start = std::chrono::high_resolution_clock::now();

    sch //.exact_copy(A(i, j, m, o), B(m, o, k, l))
      (C(j, i, k, l) += A(i, j, m, o) * B(m, o, k, l))
        .execute(ex_hw, profile);

    const auto timer_end = std::chrono::high_resolution_clock::now();

    auto mult_time =
      std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

    if(sch.ec().print())
      std::cout << "4-D Tensor contraction (C=CxC) with " << N << " indices tiled with " << tilesize
                << " : " << mult_time << std::endl;

    norm_check(C, N == 50);
    sch.deallocate(A, B, C).execute();
  }
}

template<typename T>
void test_4_dim_mult_op_last_unit(Scheduler& sch, size_t N, Tile tilesize, ExecutionHW ex_hw,
                                  bool profile) {
  TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};
  size_t          size = N / 10 > 0 ? N / 10 : 1;
  TiledIndexSpace tis2{IndexSpace{range(size)}};

  auto [i, j, k, l] = tis1.labels<4>("all");
  auto [m, o]       = tis2.labels<2>("all");

  Tensor<T> A{m, o};
  Tensor<T> B{i, j, k, m};
  Tensor<T> C{i, j, k, m};

  sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

  const auto timer_start = std::chrono::high_resolution_clock::now();

  sch(C(j, i, k, o) += A(m, o) * B(i, j, k, m)).execute(ex_hw, profile);

  const auto timer_end = std::chrono::high_resolution_clock::now();

  auto mult_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

  if(sch.ec().print())
    std::cout << "4-D Tensor contraction with 2-D unit tiled matrix " << N << " indices tiled with "
              << tilesize << " last index unit tiled : " << mult_time << std::endl;
}

template<typename T>
void test_4_dim_mult_op_first_unit(Scheduler& sch, size_t N, Tile tilesize, ExecutionHW ex_hw,
                                   bool profile) {
  TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};
  size_t          size = N / 10 > 0 ? N / 10 : 1;
  TiledIndexSpace tis2{IndexSpace{range(size)}};
  TiledIndexSpace tis3{IndexSpace{range(tilesize)}, tilesize};

  auto [i, j, k, l, m, o] = tis1.labels<6>("all");
  auto [t1, t2]           = tis3.labels<2>("all");
  // auto [m, o] = tis2.labels<2>("all");

  Tensor<T> A{m, t1};
  Tensor<T> B{t1, m};
  Tensor<T> C{m, i};

  sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();

  const auto timer_start = std::chrono::high_resolution_clock::now();

  sch(C(m, i) += A(m, t1) * B(t1, i)).execute(ex_hw, profile);

  const auto timer_end = std::chrono::high_resolution_clock::now();

  auto mult_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

  if(sch.ec().print())
    std::cout << "4-D Tensor contraction with 2-D unit tiled matrix " << N << " indices tiled with "
              << tilesize << " first index unit tiled : " << mult_time << std::endl;
}

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  if(argc < 3) { tamm_terminate("Please provide an index space size and tile size"); }

  size_t is_size   = atoi(argv[1]);
  Tile   tile_size = atoi(argv[2]);

  if(is_size < tile_size) tile_size = is_size;

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  //ExecutionHW ex_hw = ec.exhw();
  ExecutionHW ex_hw = ExecutionHW::CPU_SPARSE;

  Scheduler sch{ec};

  if(ec.print()) {
    std::cout << tamm_git_info() << std::endl;
    auto current_time   = std::chrono::system_clock::now();
    auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
    auto cur_local_time = localtime(&current_time_t);
    std::cout << std::endl << "date: " << std::put_time(cur_local_time, "%c") << std::endl;

    std::cout << "nnodes: " << ec.nnodes() << ", ";
    std::cout << "nproc_per_node: " << ec.ppn() << ", ";
    std::cout << "nproc_total: " << ec.nnodes() * ec.ppn() << ", ";
    std::cout << "ngpus_per_node: " << ec.gpn() << ", ";
    std::cout << "ngpus_total: " << ec.nnodes() * ec.gpn() << std::endl;
    std::cout << "dim, tile sizes = " << is_size << ", " << tile_size << std::endl;
    ec.print_mem_info();
    std::cout << std::endl << std::endl;
  }

  const bool profile = true;
  //test_2_dim_mult_op<double>(sch, is_size, tile_size, ex_hw, profile);
  test_3_dim_mult_op<double>(sch, is_size, tile_size, ex_hw, profile);
  //test_4_dim_mult_op<double>(sch, is_size, tile_size, ex_hw, profile);
  // test_4_dim_mult_op_last_unit<double>(sch, is_size, tile_size);
  // test_4_dim_mult_op_first_unit<double>(sch, is_size, tile_size);

  if(profile && ec.print()) {
    std::string profile_csv =
      "multops_profile_" + std::to_string(is_size) + "_" + std::to_string(tile_size) + ".csv";
    std::ofstream pds(profile_csv, std::ios::out);
    if(!pds) std::cerr << "Error opening file " << profile_csv << std::endl;
    pds << ec.get_profile_header() << std::endl;
    pds << ec.get_profile_data().str() << std::endl;
    pds.close();
  }

  tamm::finalize();

  return 0;
}
