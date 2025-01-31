#include <chrono>
#include <tamm/tamm.hpp>

using namespace tamm;

template<typename T>
void test_utils(Scheduler& sch, ExecutionHW ex_hw) {
  const int N        = 10;
  const int tilesize = 5;

  using CT = std::complex<T>;

  // get_scalar
  {
    Tensor<T>  scalar_t{};
    Tensor<CT> scalar_ct{};
    sch.allocate(scalar_t, scalar_ct).execute();
    auto val_t  = tamm::get_scalar(scalar_t);
    auto val_ct = tamm::get_scalar(scalar_ct);
    sch.deallocate(scalar_t, scalar_ct).execute();
  }

  TiledIndexSpace tis1{IndexSpace{range(N)}, tilesize};
  auto [i, j, k, l, m, o] = tis1.labels<6>("all");

  Tensor<T> imat = tamm::identity_matrix<T>(sch.ec(), tis1);
  // print_tensor(imat);
  sch.deallocate(imat);

  Tensor<CT> A{i, j, m, o}; // complex
  Tensor<T>  B{m, o, k, l}; // real
  Tensor<CT> C{i, j, k, l}; // complex

  sch.allocate(A, B, C)(A() = 21.0)(B() = 2.0)(C() = 0.0).execute();
  // C = C x R
  sch(C(j, i, k, l) += A(i, j, m, o) * B(m, o, k, l)).execute(ex_hw);

  ExecutionContext ec_dense{sch.ec().pg(), DistributionKind::dense, MemoryManagerKind::ga};

  // utils covered by other unit tests
  // - IO utils
  // - block-cyclic utils
  // - Index spaces,labels utils - union_lbl, compose_lbl, invert_lbl,
  //   project_lbl, project_tis, invert_tis, intersect_tis, union_tis

  // TODO
  // print routines (inc. print_dense_tensor)
  // fill routines: fill_tensor,fill_sparse_tensor
  // get_agg_info, subcomm_from_subranks

  {
    // operations only on 2D tensors
    Tensor<CT> ctens{i, j}; // complex
    Tensor<T>  rtens{i, j}; // real

    sch.allocate(ctens, rtens).execute();
    tamm::random_ip(rtens);

    // trace
    T  rtrace = tamm::trace(rtens);
    CT ctrace = tamm::trace(ctens);
    // trace_sqr
    T  rtrace_sqr = tamm::trace_sqr(rtens);
    CT ctrace_sqr = tamm::trace_sqr(ctens);
    // diagonal
    auto rtens_diag = tamm::diagonal(rtens);
    auto ctens_diag = tamm::diagonal(ctens);
    // update_diagonal
    tamm::update_diagonal(rtens, rtens_diag);
    tamm::update_diagonal(ctens, ctens_diag);

    sch.deallocate(ctens, rtens).execute();
  }

  // sum
  T  b_sum = tamm::sum(B);
  CT c_sum = tamm::sum(C);

  // norm
  T  b_norm = tamm::norm(B);
  CT c_norm = tamm::norm(C);

  // linf_norm
  T  b_linf_norm = tamm::linf_norm(B);
  CT c_linf_norm = tamm::linf_norm(C);

  // sqrt
  Tensor<T>  b_sqrt = tamm::sqrt(B);
  Tensor<CT> c_sqrt = tamm::sqrt(C);
  sch.deallocate(b_sqrt, c_sqrt).execute();

  // square
  Tensor<T>  b_square = tamm::square(B);
  Tensor<CT> c_square = tamm::square(C);
  sch.deallocate(b_square, c_square).execute();

  // conj, conj_ip
  Tensor<CT> c_conj = tamm::conj(C);
  tamm::conj_ip(c_conj);
  sch.deallocate(c_conj).execute();

  // pow
  Tensor<T>  b_pow = tamm::pow(B, (T) 2.0);
  Tensor<CT> c_pow = tamm::pow(C, (CT) 2.0);
  sch.deallocate(b_pow, c_pow).execute();

  // log
  Tensor<T>  b_log = tamm::log(B);
  Tensor<CT> c_log = tamm::log(C);
  sch.deallocate(b_log, c_log).execute();

  // log10
  Tensor<T>  b_log10 = tamm::log10(B);
  Tensor<CT> c_log10 = tamm::log10(C);
  sch.deallocate(b_log10, c_log10).execute();

  // einverse
  Tensor<T>  b_einverse = tamm::einverse(B);
  Tensor<CT> c_einverse = tamm::einverse(C);
  sch.deallocate(b_einverse, c_einverse).execute();

  // scale, scale_ip
  Tensor<T>  b_scale = tamm::scale(B, static_cast<T>(2.0));
  Tensor<CT> c_scale = tamm::scale(C, (CT) 2.0);
  tamm::scale_ip(b_scale, static_cast<T>(2.0));
  tamm::scale_ip(c_scale, (CT) 2.0);
  sch.deallocate(b_scale, c_scale).execute();

  // random_ip
  tamm::random_ip(A);
  tamm::random_ip(B);

  // max_element,min_element
  auto [bmaxval, bmaxblockid, bmaxcoord] = tamm::max_element(B);
  auto [bminval, bminblockid, bmincoord] = tamm::min_element(B);

  // update_tensor_val
  const T                   val_t = 42.0;
  const CT                  val_ct(2, 3);
  const std::vector<size_t> t_coord = {1, 2, 4, 1};
  update_tensor_val(B, t_coord, val_t);
  update_tensor_val(C, t_coord, val_ct);

  // hash_tensor
  auto b_hash = tamm::hash_tensor(B);
  auto c_hash = tamm::hash_tensor(C);

  // permute_tensor
  Tensor<T>  b_perm = permute_tensor<T>(B, {1, 3, 2, 0});
  Tensor<CT> c_perm = permute_tensor<CT>(C, {1, 3, 2, 0});
  sch.deallocate(b_perm, c_perm).execute();

  // to_dense_tensor
  Tensor<T>  b_dens = tamm::to_dense_tensor(ec_dense, B);
  Tensor<CT> c_dens = tamm::to_dense_tensor(ec_dense, C);

  // get_tensor_element
  T  b_dens_val = tamm::get_tensor_element(b_dens, {0, 0, 0, 0});
  CT c_dens_val = tamm::get_tensor_element(c_dens, {0, 0, 0, 0});

  // tensor_block
  Tensor<T>  b_block = tamm::tensor_block(b_dens, {0, 0, 0, 0}, {N / 2, N / 2, N / 2, N / 2});
  Tensor<CT> c_block = tamm::tensor_block(c_dens, {0, 0, 0, 0}, {N / 2, N / 2, N / 2, N / 2});

  // local_buf_size and access_local_buf ... to be moved to a unit test
  if(int(std::pow(N / tilesize, b_dens.num_modes())) % ec_dense.pg().size().value() == 0)
    EXPECTS(std::pow(N, b_dens.num_modes()) / ec_dense.pg().size().value() ==
            b_dens.local_buf_size());
  if(ec_dense.pg().rank() == 0)
    EXPECTS(*b_dens.access_local_buf() == tamm::get_tensor_element(b_dens, {0, 0, 0, 0}));

  sch.deallocate(b_dens, c_dens).execute();
  sch.deallocate(b_block, c_block).execute();

  // redistribute_tensor
  TiledIndexSpace    tis_red{IndexSpace{range(N)}, N / 2};
  TiledIndexSpaceVec tis_red_vec{tis_red, tis_red, tis_red, tis_red};
  Tensor<T>          b_red = tamm::redistribute_tensor<T>(B, tis_red_vec);
  Tensor<CT>         c_red = tamm::redistribute_tensor<CT>(C, tis_red_vec);
  sch.deallocate(b_red, c_red).execute();

  // retile_tamm_tensor
  Tensor<T>  b_ret{tis_red, tis_red, tis_red, tis_red};
  Tensor<CT> c_ret{tis_red, tis_red, tis_red, tis_red};
  sch.allocate(b_ret, c_ret).execute();
  tamm::retile_tamm_tensor(B, b_ret);
  tamm::retile_tamm_tensor(C, c_ret);

  sch.deallocate(b_ret, c_ret).execute();
  sch.deallocate(A, B, C).execute();
}

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  ExecutionHW ex_hw = ec.exhw();

  Scheduler sch{ec};

  test_utils<double>(sch, ex_hw);

  tamm::finalize();

  return 0;
}
