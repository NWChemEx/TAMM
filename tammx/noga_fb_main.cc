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

std::vector <Spin> spins = {1_sp, -1 * 1_sp, 1_sp, -1 * 1_sp};
std::vector <Irrep> spatials = {0_ir, 0_ir, 0_ir, 0_ir};
std::vector <size_t> sizes = {2, 4, 2, 1};
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

void test() {
  using Type = Tensor::Type;
  using Distribution = Tensor::Distribution;

  TensorVec <SymmGroup> indices_ov{SymmGroup{DimType::o}, SymmGroup{DimType::v}};
  TensorVec <SymmGroup> indices_vn{SymmGroup{DimType::v}, SymmGroup{DimType::n}};
  TensorVec <SymmGroup> indices_on{SymmGroup{DimType::o}, SymmGroup{DimType::n}};
  TensorVec <SymmGroup> indices_oo{SymmGroup{DimType::o}, SymmGroup{DimType::o}};
  TensorVec <SymmGroup> indices_vv{SymmGroup{DimType::v}, SymmGroup{DimType::v}};
  TensorVec <SymmGroup> indices_nn{SymmGroup{DimType::n}, SymmGroup{DimType::n}};
  TensorVec <SymmGroup> t_scalar{};

  Tensor bDiag{t_scalar, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor t1{t_scalar, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor t2{t_scalar, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor t3{t_scalar, Type::double_precision, Distribution::tce_nwma, 0, irrep_t, false};

  Tensor t4{indices_on, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor t5{indices_on, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor t6{indices_on, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor t7{indices_vn, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};

  Tensor FT{indices_nn, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor hT{indices_nn, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor bT{indices_nn, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor X_OO{indices_oo, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor X_OV{indices_ov, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor X_VV{indices_vv, Type::double_precision, Distribution::tce_nwma, 1, irrep_t, false};

  FT.allocate();
  hT.allocate();
  bT.allocate();
  bDiag.allocate();

  X_OO.allocate();
  X_OV.allocate();
  X_VV.allocate();

  t1.allocate();
  t2.allocate();
  t3.allocate();
  t4.allocate();
  t5.allocate();
  t6.allocate();
  t7.allocate();


  tensor_map(hT(), [](Block &block) {
    //std::fill_n(reinterpret_cast<double*>(block.buf()), block.size(), 1.0);
    int n = 0;
    //std::generate_n(reinterpret_cast<double*>(block.buf()), block.size(), std::rand);
    std::generate_n(reinterpret_cast<double *>(block.buf()), block.size(), [&]() { return n++; });
  });

  tensor_map(bT(), [](Block &block) {
    int n = 0;
    std::generate_n(reinterpret_cast<double *>(block.buf()), block.size(), [&]() { return n++; });
  });

  VLabel a{0}, b{1};
  OLabel i{0}, j{1};
  NLabel p{0}, q{1};

  //hf_1: FT[p,q] += 1.0 * hT[p,q];
  FT() += 1.0 * hT();

  //hf_2:  FT[p,q] += bDiag * bT[p,q];
  FT() += bDiag() * bT();

  //hf_3_1: t1 += X_OO[i,j] * bT[i,j];
  t1() += X_OO({i,j}) * bT({i,j});
  //hf_3: FT[p,q] += bT[p,q] * t1;
  FT() += bT() * t1();

  //hf_4_1: t2 += 2.0 * X_OV[i,a] * bT[i,a];
  t2() += 2.0 * X_OV({i,a}) * bT({i,a});
  //hf_4:  FT[p,q] += t2 * bT[p,q];
  FT() += t2() * bT();

  //hf_5_1: t3 += X_VV[a,b] * bT[a,b];
  t3() += X_VV({a,b}) * bT({a,b});
  //hf_5: FT[p,q] += bT[p,q] * t3;
  FT() += bT() * t3();

  //hf_6: FT[p,q] += -1.0 * bT[p,i] * bT[i,q];
  //FT({p0,p2}) += -1.0 * bT({p0,p1}) * bT({p1,p2});
  FT({p,q}) += -1.0 * bT({p,i}) * bT({i,q});

  //hf_7_1: t4[i,q] += X_OO[i,j] * bT[j,q];
  t4({i,q}) += X_OO({i,j}) * bT({j,q});

  //hf_7:  FT[p,q] += -1.0 *  bT[p,i] * t4[i,q];
  FT({p,q}) += -1.0 * bT({p,i}) * t4({i,q});

  // hf_8_1: t5[i,q] += X_OV[i,a] * bT[a,q];
  t5({i,q}) += X_OV({i,a}) * bT({a,q});
  // hf_8:  FT[p,q] += -1.0 * bT[p,i] * t5[i,q];
  FT({p,q}) += -1.0 * bT({p,i}) * t5({i,q});

  //hf_9_1: t6[i,p] += X_OV[i,a] * bT[p,a];
  t6({i,p}) += X_OV({i,a}) * bT({p,a});
  //hf_9: FT[p,q] += -1.0 * bT[i,q] * t6[i,p];
  FT({p,q}) += -1.0 * bT({i,q}) * t6({i,p});

  //hf_10_1: t7[a,q] += X_VV[a,b] * bT[b,q];
  t7({a,q}) += X_VV({a,b}) * bT({b,q});

  //hf_10:  FT[p,q] += -1.0 * bT[p,a] * t7[a,q];
  FT({p,q}) += -1.0 * bT({p,a}) * t7({a,q});

  std::cerr << "------------------" << std::endl;
  tensor_print(FT, std::cerr);
  std::cerr << "------------------" << std::endl;

//  std::cerr << "------------------" << std::endl;
//  tensor_print(hT, std::cerr);
//  std::cerr << "------------------" << std::endl;

  FT.destruct();
  hT.destruct();
  bT.destruct();
  bDiag.destruct();

  X_OO.destruct();
  X_OV.destruct();
  X_VV.destruct();

  t1.destruct();
  t2.destruct();
  t3.destruct();
  t4.destruct();
  t5.destruct();
  t6.destruct();
  t7.destruct();
}

void density_test(Tensor* D, Tensor* F) {
  Tensor X, T, R, Z; //, delta;

  for (int l1=0;l1<20;l1++) {
    for (int l2 = 0;l2< 20;l2++){
      Tensor tmp1, tmp2, tmp3;

      R({i,a}) += 1.0 * F({i,a});
      R({i,a}) += -1.0 * (F({i,b}) * X({b,a}));
      R({i,a}) += -1.0 * (F({i,m}) * X({m,a}));
      R({i,a}) += X({i,j}) * F({j,a});
      R({i,a}) += -1.0 * (X({i,j}) * F({j,b}));
      R({i,a}) += 1.0 * X({b,a});

      tmp1({j,a}) += 1.0 * (F({j,m}) * X({m,a}));
      R({i,a}) += -1.0 * (X({i,j}) * tmp1({j,a}));
      R({i,a}) += 1.0 * (X({i,b}) * F({b,a}));

      tmp2({b,a}) += 1.0 * (F({b,c}) * X({c,a}));
      R({i,a}) += -1.0 * (X({i,b}) * tmp2({b,a}));
      tmp3({b,a}) += 1.0 * (F({b,m}) * X({m,a}));
      R({i,a}) += -1.0 * (X({i,b}) * tmp3({b,a}));


      // T({i,a}) += Z({i,a}) * R({i,a});  // div (f_aa(i) - f_ii(i))
      // T({i,a}) += T({i,a}) / (F({a,a}) - F({i,i}));
    }

    Z({i,j}) += -1.0 * (T({i,e}) * T({j,e}));

    // following loop solves this equation:
    // r_ij = D_ij - delta_ij - D_im * Z_mj = 0
    for (int l3=0;l3<10;l3++){
      D({i,j}) += 1.0 * delta({i,j});
      D({i,j}) += 1.0 * (D({i,m}) * Z({m,j}));
      // D({i,j}) += R({i,j}) / (delta({i,j}) + Z({i,j}));
    }

    X({i,a}) += 1.0 * (D({i,m}) * T({m,a}));

    X({a,b}) += 1.0 * (X({m,a}) * T({m,b}));

    X({i,j}) += -1.0 * (T({i,e}) * X({j,e}));

    // call fock build

    // notes follow
    //X({i,a}) = X({a,i})

    // #Construct new fock matrix F(D(i+1)) from Density matrix D
    // #X_ia,X_ab,X_ij (i+1)

    // full D(i+1) = [ l_oo + x_oo | X_ov ]
    //               [ X_vo        | X_vv ]

    // #D_ij = [delta_ij + X_ij ]

  }
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

  test();

  TCE::finalize();
  return 0;
}
