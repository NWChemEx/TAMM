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
  return Irrep{strongint_cast<Irrep::value_type>(val)};
}

Spin operator "" _sp(unsigned long long int val) {
  return Spin{strongint_cast<Spin::value_type>(val)};
}

BlockDim operator "" _bd(unsigned long long int val) {
  return BlockDim{strongint_cast<BlockDim::value_type>(val)};
}

std::vector<Spin> spins = {1_sp, -1 * 1_sp, 1_sp, -1 * 1_sp};
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

void noga_fock_build() { /// @todo pass F, X_OO,X_OV,X_VV
  using Type = Tensor::Type;
  using Distribution = Tensor::Distribution;

  TensorVec<SymmGroup> indices_ov{SymmGroup{DimType::o}, SymmGroup{DimType::v}};
  TensorVec<SymmGroup> indices_vn{SymmGroup{DimType::v}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> indices_on{SymmGroup{DimType::o}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> indices_oo{SymmGroup{DimType::o}, SymmGroup{DimType::o}};
  TensorVec<SymmGroup> indices_vv{SymmGroup{DimType::v}, SymmGroup{DimType::v}};
  TensorVec<SymmGroup> indices_nn{SymmGroup{DimType::n}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> t_scalar{};

  /// @todo FIXME: bDiag is the diagonal vector, not a scalar
  Tensor bDiag{t_scalar, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor t1{t_scalar, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor t2{t_scalar, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor t3{t_scalar, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};

  Tensor t4{indices_on, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor t5{indices_on, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor t6{indices_on, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor t7{indices_vn, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};

  Tensor hT{indices_nn, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor bT{indices_nn, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};

  hT.allocate();
  bT.allocate();
  bDiag.allocate();
  t1.allocate();
  t2.allocate();
  t3.allocate();
  t4.allocate();
  t5.allocate();
  t6.allocate();
  t7.allocate();

  /// @todo F, X_.. tensors allocation code should go away since they are passed from noga_main()
  Tensor FT{indices_nn, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor X_OO{indices_oo, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor X_OV{indices_ov, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor X_VV{indices_vv, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};

  FT.allocate();
  X_OO.allocate();
  X_OV.allocate();
  X_VV.allocate();


  tensor_map(hT(), [](Block &block) {
    int n = 0;
    std::generate_n(reinterpret_cast<double *>(block.buf()), block.size(), [&]() { return n++; });
  });

  tensor_map(bT(), [](Block &block) {
    int n = 0;
    std::generate_n(reinterpret_cast<double *>(block.buf()), block.size(), [&]() { return n++; });
  });

  VLabel a{0}, b{1};
  OLabel i{0}, j{1};
  NLabel p{0}, q{1};

  //FT=F,X_OO,X_OV,X_VV come from noga_main(...)
  /// @todo tensor_init(FT,0.0); FT=0 each time the fock build is started
  /// or simply do FT({p, q}) = 1.0 * hT({p, q}); once = is implemented
  FT({p, q}) += 1.0 * hT({p, q});

  /// @todo loop over Q for all code below

  FT({p, q}) += bDiag() * bT({p, q});

  t1() += X_OO({i, j}) * bT({i, j});
  FT({p, q}) += bT({p, q}) * t1();

  t2() += 2.0 * X_OV({i, a}) * bT({i, a});
  bT({p, q}) += t2() * bT({p, q});

  t3() += X_VV({a, b}) * bT({a, b});
  bT({p, q}) += bT({p, q}) * t3();

  FT({p, q}) += -1.0 * bT({p, i}) * bT({i, q});

  t4({i, q}) += X_OO({i, j}) * bT({j, q});
  FT({p, q}) += -1.0 * bT({p, i}) * t4({i, q});

  t5({i, q}) += X_OV({i, a}) * bT({a, q});
  FT({p, q}) += -1.0 * bT({p, i}) * t5({i, q});

  t6({i, p}) += X_OV({i, a}) * bT({p, a});
  FT({p, q}) += -1.0 * bT({i, q}) * t6({i, p});

  t7({a, q}) += X_VV({a, b}) * bT({b, q});
  FT({p, q}) += -1.0 * bT({p, a}) * t7({a, q});

  std::cerr << "------------------" << std::endl;
  tensor_print(FT, std::cerr);
  std::cerr << "------------------" << std::endl;

  hT.destruct();
  bT.destruct();
  bDiag.destruct();
  t1.destruct();
  t2.destruct();
  t3.destruct();
  t4.destruct();
  t5.destruct();
  t6.destruct();
  t7.destruct();

  /// @todo FT, X_.. destruct go away since they are passed from noga_main()
  FT.destruct();
  X_OO.destruct();
  X_OV.destruct();
  X_VV.destruct();
}

void noga_main() {

  using Type = Tensor::Type;
  using Distribution = Tensor::Distribution;

  TensorVec<SymmGroup> indices_ov{SymmGroup{DimType::o}, SymmGroup{DimType::v}};
  TensorVec<SymmGroup> indices_vn{SymmGroup{DimType::v}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> indices_on{SymmGroup{DimType::o}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> indices_oo{SymmGroup{DimType::o}, SymmGroup{DimType::o}};
  TensorVec<SymmGroup> indices_vv{SymmGroup{DimType::v}, SymmGroup{DimType::v}};
  TensorVec<SymmGroup> indices_nn{SymmGroup{DimType::n}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> t_scalar{};

  /// @todo Tensor D,F come from env which is noga_main for now.
  Tensor D{indices_oo, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor F{indices_nn, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};

  Tensor delta{indices_oo, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor R1{indices_ov, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor R2{indices_oo, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor T{indices_ov, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor Z{indices_oo, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};

  Tensor X_OO{indices_oo, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor X_OV{indices_ov, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor X_VV{indices_vv, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};

  Tensor tmp1{t_scalar, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor tmp2{t_scalar, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor tmp3{t_scalar, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor tmp0{indices_ov, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor tmp4{indices_oo, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor tmp5{indices_oo, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};

  VLabel a{0}, b{1}, c{2}, d{3}, e{4};
  OLabel i{0}, j{1}, m{2};
  NLabel p{0}, q{1};

  D.allocate();
  F.allocate();
  R1.allocate();
  R2.allocate();
  T.allocate();
  Z.allocate();
  delta.allocate();

  tmp0.allocate();
  tmp1.allocate();
  tmp2.allocate();
  tmp3.allocate();
  tmp4.allocate();
  tmp5.allocate();

  X_OO.allocate();
  X_OV.allocate();
  X_VV.allocate();

  /// @todo tensor_init(T,0)
  /// @todo tensor_init(D,0)
  /// @todo T,D are initialized only once at the start and always accumulated in the rest of the code

  for (int l1 = 0; l1 < 20; l1++) { //OUTERMOST LOOP - LOOP1


    for (int l2 = 0; l2 < 20; l2++) { /// LOOP2
      /// @todo tensor_init(R1,0); R1=0 for each iteration of L2

      R1({i, a}) += 1.0 * F({i, a});
      R1({i, a}) += -1.0 * (F({i, b}) * X_VV({b, a}));
      R1({i, a}) += -1.0 * (F({i, m}) * X_OV({m, a}));
      R1({i, a}) += X_OO({i, j}) * F({j, a});
      /// @todo tensor_init(tmp0,0)
      tmp0({i, b}) += -1.0 * (X_OO({i, j}) * F({j, b}));
      R1({i, a}) += 1.0 * tmp0({i, b}) * X_VV({b, a});

      /// @todo tensor_init(tmp1,0)
      tmp1({j, a}) += 1.0 * (F({j, m}) * X_OV({m, a}));
      R1({i, a}) += -1.0 * (X_OO({i, j}) * tmp1({j, a}));
      R1({i, a}) += 1.0 * (X_OV({i, b}) * F({b, a}));

      /// @todo tensor_init(tmp2,0)
      tmp2({b, a}) += 1.0 * (F({b, c}) * X_VV({c, a}));
      R1({i, a}) += -1.0 * (X_OV({i, b}) * tmp2({b, a}));

      /// @todo tensor_init(tmp3,0)
      tmp3({b, a}) += 1.0 * (F({b, m}) * X_OV({m, a}));
      R1({i, a}) += -1.0 * (X_OV({i, b}) * tmp3({b, a}));

      /// @todo Denominator can be a scalar for now
      //T({i,a}) += R1({i,a}) / (F({a,a}) - F({i,i}));

    }

    /// @todo tensor_init(Z,0); Z=0 for each iteration of L1
    Z({i, j}) += -1.0 * (T({i, e}) * T({j, e}));

    // following loop solves this equation:
    // r_ij = D_ij - delta_ij - D_im * Z_mj
    // D_ij = R_ij / (delta_ij + Z_ij)
    /// NOTE: delta({i,j}) is a Unit matrix: 0 for i!=j 1 for i=j

    for (int l3 = 0; l3 < 10; l3++) {  // LOOP 3
      /// @todo tensor_init(R2,0); R2=0 for each iteration of L3
      tmp4({i, j}) += 1.0 * (D({i, m}) * Z({m, j}));
      /// @todo Uncomment the following 4 lines once the logic is implemented
      /// tensor_init(t5,0);
      /// tmp5({i,j}) += delta({i, j}) - tmp4({i, j});
      /// R2({i, j}) += D({i, j}) - tmp5({i, j});
      /// D({i, j}) += R2({i, j}) / (delta({i, j}) + Z({i, j}));
    }

    /// @todo tensor_initialize X_OV=0, X_OO=0, X_VV=0
    /// X tensors are newly created in every iteration of L1 and passed to fock build
    X_OV({i, a}) += 1.0 * (D({i, m}) * T({m, a}));
    X_VV({a, b}) += 1.0 * (X_OV({m, a}) * T({m, b}));
    X_OO({i, j}) += -1.0 * (T({i, e}) * X_OV({j, e}));

    /// call fock build
    /// @todo noga_fock_build(F, X_OV, X_VV, X_OO);
    noga_fock_build();

  }

  D.destruct();
  F.destruct();
  R1.destruct();
  R2.destruct();
  T.destruct();
  Z.destruct();
  delta.destruct();

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

  noga_main();

  TCE::finalize();
  return 0;
}
