#include <vector>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <string>

#undef NDEBUG

#include "tammx/tammx.h"


using namespace std;
using namespace tammx;

Irrep operator "" _ir(unsigned long long int val) {
  return Irrep{checked_cast<Irrep::value_type>(val)};
}

Spin operator "" _sp(unsigned long long int val) {
  return Spin{checked_cast<Spin::value_type>(val)};
}

BlockDim operator "" _bd(unsigned long long int val) {
  return BlockDim{checked_cast<BlockDim::value_type>(val)};
}

std::vector<Spin> spins = {1_sp, 2_sp, 2_sp, 2_sp};
std::vector<Irrep> spatials = {0_ir, 0_ir, 0_ir, 0_ir};
std::vector<size_t> sizes = {2, 4, 2, 1};
BlockDim noa{1};
BlockDim noab{2};
BlockDim nva{1};
BlockDim nvab{2};
bool spin_restricted = false;
Irrep irrep_f{0};
Irrep irrep_v{0};
Irrep irrep_t{0};
Irrep irrep_x{0};
Irrep irrep_y{0};

struct OLabel : public IndexLabel {
  OLabel(int n)
      : IndexLabel{n, DimType::o} {}
};

struct VLabel : public IndexLabel {
  VLabel(int n)
      : IndexLabel{n, DimType::v} {}
};

struct NLabel : public IndexLabel {
  NLabel(int n)
      : IndexLabel{n, DimType::n} {}
};

void noga_fock_build(Tensor& F, Tensor& X_OV,
                     Tensor& X_VV, Tensor& X_OO,
                     Tensor& hT, Tensor& bT, double bdiagsum) {
  auto eltype = F.element_type();
  auto distribution = Tensor::Distribution::tce_nwma;

  TensorVec<SymmGroup> indices_ov{SymmGroup{DimType::o}, SymmGroup{DimType::v}};
  TensorVec<SymmGroup> indices_vn{SymmGroup{DimType::v}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> indices_on{SymmGroup{DimType::o}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> indices_oo{SymmGroup{DimType::o}, SymmGroup{DimType::o}};
  TensorVec<SymmGroup> indices_vv{SymmGroup{DimType::v}, SymmGroup{DimType::v}};
  TensorVec<SymmGroup> indices_nn{SymmGroup{DimType::n}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> t_scalar{};

  Tensor t1{t_scalar, eltype, distribution, 0, irrep_t, false};
  Tensor t2{t_scalar, eltype, distribution, 0, irrep_t, false};
  Tensor t3{t_scalar, eltype, distribution, 0, irrep_t, false};

  Tensor t4{indices_on, eltype, distribution, 1, irrep_t, false};
  Tensor t5{indices_on, eltype, distribution, 1, irrep_t, false};
  Tensor t6{indices_on, eltype, distribution, 1, irrep_t, false};
  Tensor t7{indices_vn, eltype, distribution, 1, irrep_t, false};

  Tensor FT{indices_nn, eltype, distribution, 1, irrep_t, false};

  t1.allocate();
  t2.allocate();
  t3.allocate();
  t4.allocate();
  t5.allocate();
  t6.allocate();
  t7.allocate();
  FT.allocate();

  VLabel a{0}, b{1};
  OLabel i{0}, j{1};
  NLabel p{0}, q{1};

  OpList()
      (FT(p, q) = 1.0 * hT(p, q))
      .execute();

  /// @todo loop over Q for all code below

  OpList()
      (FT(p, q) +=        bdiagsum   * bT(p, q))
      (t1()      =        X_OO(i, j) * bT(i, j))
      (FT(p, q) +=        bT(p, q)   * t1())
      (t2()      = 2.0 *  X_OV(i, a) * bT(i, a))
      (bT(p, q) +=        t2()       * bT(p, q))
      (t3()      =        X_VV(a, b) * bT(a, b))
      (bT(p, q) +=        bT(p, q)   * t3())
      (FT(p, q) += -1.0 * bT(p, i)   * bT(i, q))
      (t4(i, q)  =        X_OO(i, j) * bT(j, q))
      (FT(p, q) += -1.0 * bT(p, i)   * t4(i, q))
      (t5(i, q)  =        X_OV(i, a) * bT(a, q))
      (FT(p, q) += -1.0 * bT(p, i)   * t5(i, q))
      (t6(i, p)  =        X_OV(i, a) * bT(p, a))
      (FT(p, q) += -1.0 * bT(i, q)   * t6(i, p))
      (t7(a, q)  =        X_VV(a, b) * bT(b, q))
      (FT(p, q) += -1.0 * bT(p, a)   * t7(a, q))
      .execute();

  t1.destruct();
  t2.destruct();
  t3.destruct();
  t4.destruct();
  t5.destruct();
  t6.destruct();
  t7.destruct();

  FT.destruct();
}

void extract_diag(Tensor& F, double* fdiag) {
  int pos = 0;
  OpList()
      .sop<2>(F(), [&pos,&fdiag] (auto p, auto q, auto& val) {
          if(p == q) {
            fdiag[pos++] = val;
          }
        })
      .execute();
}

void noga_main(Tensor& D, Tensor& F, Tensor& hT, Tensor& bT, double bdiagsum) {
  auto distribution = D.distribution();
  auto eltype = D.element_type();

  TensorVec<SymmGroup> indices_ov{SymmGroup{DimType::o}, SymmGroup{DimType::v}};
  TensorVec<SymmGroup> indices_vn{SymmGroup{DimType::v}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> indices_on{SymmGroup{DimType::o}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> indices_oo{SymmGroup{DimType::o}, SymmGroup{DimType::o}};
  TensorVec<SymmGroup> indices_vv{SymmGroup{DimType::v}, SymmGroup{DimType::v}};
  TensorVec<SymmGroup> indices_nn{SymmGroup{DimType::n}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> t_scalar{};

  Tensor R1{indices_ov, eltype, distribution, 1, irrep_t, false};
  Tensor R2{indices_oo, eltype, distribution, 1, irrep_t, false};
  Tensor T{indices_ov,  eltype, distribution, 1, irrep_t, false};
  Tensor Z{indices_oo,  eltype, distribution, 1, irrep_t, false};

  Tensor X_OO{indices_oo, eltype, distribution, 1, irrep_t, false};
  Tensor X_OV{indices_ov, eltype, distribution, 1, irrep_t, false};
  Tensor X_VV{indices_vv, eltype, distribution, 1, irrep_t, false};

  Tensor tmp1{indices_ov, eltype, distribution, 1, irrep_t, false};
  Tensor tmp2{indices_vv, eltype, distribution, 1, irrep_t, false};
  Tensor tmp3{indices_vv, eltype, distribution, 1, irrep_t, false};
  Tensor tmp0{indices_ov, eltype, distribution, 1, irrep_t, false};
  Tensor tmp4{indices_oo, eltype, distribution, 1, irrep_t, false};
  Tensor tmp5{indices_oo, eltype, distribution, 1, irrep_t, false};

  VLabel a{0}, b{1}, c{2}, d{3}, e{4};
  OLabel i{0}, j{1}, m{2};
  NLabel p{0}, q{1};

  R1.allocate();
  R2.allocate();
  T.allocate();
  Z.allocate();

  tmp0.allocate();
  tmp1.allocate();
  tmp2.allocate();
  tmp3.allocate();
  tmp4.allocate();
  tmp5.allocate();

  X_OO.allocate();
  X_OV.allocate();
  X_VV.allocate();

  OpList()
      (T() = 0.0)
      (D() = 0.0)
      .execute();

  double fdiag[TCE::total_dim_size()];
  extract_diag(F, fdiag);

  auto delta = [] (int i, int j) {
    return (i==j) ? 1 : 0;
  };

  for (int l1 = 0; l1 < 20; l1++) { //OUTERMOST LOOP - LOOP1

    for (int l2 = 0; l2 < 20; l2++) { /// LOOP2
      OpList()
          (R1(i, a)    =        F(i, a))
          (R1(i, a)   += -1.0 * F(i, b)    * X_VV(b, a))
          (R1(i, a)   += -1.0 * F(i, m)    * X_OV(m, a))
          (R1(i, a)   +=        X_OO(i, j) * F(j, a))
          (tmp0(i, b)  = -1.0 * X_OO(i, j) * F(j, b))
          (R1(i, a)   +=        tmp0(i, b) * X_VV(b, a))
          (tmp1(j, a)  =        F(j, m)    * X_OV(m, a))
          (R1(i, a)   += -1.0 * X_OO(i, j) * tmp1(j, a))
          (R1(i, a)   +=        X_OV(i, b) * F(b, a))
          (tmp2(b, a)  =        F(b, c)    * X_VV(c, a))
          (R1(i, a)   += -1.0 * X_OV(i, b) * tmp2(b, a))
          (tmp3(b, a)  =        F(b, m)    * X_OV(m, a))
          (R1(i, a)   += -1.0 * X_OV(i, b) * tmp3(b, a))
          .op<2,1>(T(), R1(), [&fdiag] (auto a, auto i, auto& lval, auto& r1val) {
              lval = r1val / (fdiag[a] - fdiag[i]);
            })
          .execute();
    }

    OpList::execute(Z(i, j) = -1.0 * T(i, e) * T(j, e));

    // following loop solves this equation:
    // r_ij = D_ij - delta_ij - D_im * Z_mj
    // D_ij = R_ij / (delta_ij + Z_ij)
    /// NOTE: delta({i,j}) is a Unit matrix: 0 for i!=j 1 for i=j

    for (int l3 = 0; l3 < 10; l3++) {  // LOOP 3
      OpList()
          (tmp4({i, j}) = 1.0 * D({i, m}) * Z({m, j}))
          .op<2,1>(tmp5(), tmp4(), [&delta] (auto i, auto j, auto& t5val, auto& t4val) {
              t5val = delta(i, j) + t4val;
            })
          (R2({i,j}) = D({i,j}))
          (R2({i,j}) += -1.0 * tmp5({i,j}))
          .op<2,2>(D(), R2(), Z(),
                   [&delta] (auto i, auto j, auto& dval, auto &r2val, auto &zval) {
                     dval = r2val / (delta(i,j) + zval);
                   })
          .execute();
    }

    OpList()
        (X_OV({i, a}) = 1.0 * D({i, m}) * T({m, a}))
        (X_VV({a, b}) = 1.0 * X_OV({m, a}) * T({m, b}))
        (X_OO({i, j}) = -1.0 * T({i, e}) * X_OV({j, e}))
        .execute();

#if 0
    noga_fock_build(F, X_OV, X_VV, X_OO, hT, bT, bdiagsum);
#endif
    extract_diag(F, fdiag);
  }

  R1.destruct();
  R2.destruct();
  T.destruct();
  Z.destruct();

  tmp0.destruct();
  tmp1.destruct();
  tmp2.destruct();
  tmp3.destruct();
  tmp4.destruct();
  tmp5.destruct();

  X_OO.destruct();
  X_OV.destruct();
  X_VV.destruct();
}

void noga_driver() {
  auto eltype = ElementType::double_precision;
  auto distribution = Tensor::Distribution::tce_nwma;
  TensorVec<SymmGroup> indices_oo{SymmGroup{DimType::o}, SymmGroup{DimType::o}};
  TensorVec<SymmGroup> indices_nn{SymmGroup{DimType::n}, SymmGroup{DimType::n}};

  /// @todo what is the initial guess
  Tensor D{indices_oo, eltype, distribution, 1, irrep_t, false};
  /// @todo what is the initial guess
  Tensor F{indices_nn, eltype, distribution, 1, irrep_t, false};
  ///@todo comes from integrals
  Tensor bT{indices_nn, eltype, distribution, 1, irrep_t, false};

  ///@todo Who produces this?
  Tensor hT{indices_nn, eltype, distribution, 1, irrep_t, false};

  hT.allocate();
  D.allocate();
  F.allocate();
  bT.allocate();

  OpList()
      (hT(), [] (auto &ival) { ival = 1; })
      (bT(), [] (auto &ival) { ival = 2; })
      .execute();

  double bdiagsum;
  {
    std::vector<double> bdiag(TCE::total_dim_size());
    extract_diag(bT, &bdiag[0]);
    bdiagsum = 0;
    for(auto &b: bdiag) {
      bdiagsum += b;
    }
  }

  noga_main(D, F, hT, bT, bdiagsum);

  hT.destruct();
  bT.destruct();
  D.destruct();
  F.destruct();
}

void op_test() {
  auto eltype = ElementType::double_precision;
  auto distribution = Tensor::Distribution::tce_nwma;
  TensorVec<SymmGroup> indices_oo{SymmGroup{DimType::o, DimType::o}};
  TensorVec<SymmGroup> indices_o_o{SymmGroup{DimType::o}, SymmGroup{DimType::o}};
  TensorVec<SymmGroup> indices_nn{SymmGroup{DimType::n}, SymmGroup{DimType::n}};

  Tensor ta1{indices_o_o, eltype, distribution, 2, irrep_t, false};
  Tensor ta2{indices_o_o, eltype, distribution, 2, irrep_t, false};

  OLabel i{0}, j{1};
  
  ta1.allocate();
  ta2.allocate();

  OpList()
      (ta1(), [] (auto &ival) {
        auto x = rand()%10;
        std::cerr<<"ta1()., val="<<x<<std::endl;
        //ival = rand()%10;
        ival = x;
      })
      (ta2(), [] (auto &ival) { ival = 10 + (rand()%10); })
      .execute();

  tensor_print(ta1()); std::cout<<std::endl;

  OpList()
      (ta1() = ta2())
      (ta1() += -1.0 * ta2())
      .execute();
  // OpList()
  //     (ta1() = 0.0)
  //     .execute();
  // OpList()
  //     (ta1(), [] (auto& ival) {
  //       std::cerr<<"ta1(). resetting val="<<std::endl;
  //       ival = 0.0;
  //     })
  //     .execute();

  tensor_print(ta1()); std::cout<<std::endl;  
  assert_zero(ta1);

  OpList()(ta1() = 5).execute();
  assert_equal(ta1(), 5.0);

  OpList()
      (ta1() = 4)
      (ta2() = 6)
      (ta1() += ta2())
      .execute();

  assert_equal(ta1(), 10);

  OpList()
      (ta1() = 7)
      (ta2() = 9.0)
      (ta1() += -1 * ta2())
      .execute();
  assert_equal(ta1(), -2);

  OpList()
      (ta1(), [] (auto &ival) { ival = rand() % 100; })
      (ta2(i,j) = 0.5 * ta1(i,j))
      (ta2(i,j) += 0.5 * ta1(j,i))
      (ta1() = ta2())
      (ta1(i,j) += -1 * ta2(j,i))
      .execute();

  assert_zero(ta1);
  
  ta1.destruct();
  ta2.destruct();
}


int main() {
  TCE::init(spins, spatials, sizes,
            noa,
            noab,
            nva,
            nvab,
            spin_restricted,
            irrep_f,
            irrep_v,
            irrep_t,
            irrep_x,
            irrep_y);

  //noga_driver();
  op_test();
  
  TCE::finalize();
  return 0;
}


//for now

#include "tammx/block.cc"
#include "tammx/tensor.cc"
