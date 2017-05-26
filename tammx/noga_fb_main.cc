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
  using Type = ElementType;
  using Distribution = Tensor::Distribution;

  TensorVec<SymmGroup> indices_ov{SymmGroup{DimType::o}, SymmGroup{DimType::v}};
  TensorVec<SymmGroup> indices_vn{SymmGroup{DimType::v}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> indices_on{SymmGroup{DimType::o}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> indices_oo{SymmGroup{DimType::o}, SymmGroup{DimType::o}};
  TensorVec<SymmGroup> indices_vv{SymmGroup{DimType::v}, SymmGroup{DimType::v}};
  TensorVec<SymmGroup> indices_nn{SymmGroup{DimType::n}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> t_scalar{};

  Tensor t1{t_scalar, ElementType::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor t2{t_scalar, ElementType::double_precision, Distribution::tce_nwma, 0, irrep_t, false};
  Tensor t3{t_scalar, ElementType::double_precision, Distribution::tce_nwma, 0, irrep_t, false};

  Tensor t4{indices_on, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor t5{indices_on, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor t6{indices_on, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor t7{indices_vn, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};

  t1.allocate();
  t2.allocate();
  t3.allocate();
  t4.allocate();
  t5.allocate();
  t6.allocate();
  t7.allocate();

  Tensor FT{indices_nn, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  FT.allocate();

  VLabel a{0}, b{1};
  OLabel i{0}, j{1};
  NLabel p{0}, q{1};

  FT.init(0.0);
  FT({p, q}) += 1.0 * hT({p, q});

  /// @todo loop over Q for all code below

  FT({p, q}) += bdiagsum * bT({p, q});

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
  NLabel p{0}, q{1};
  block_for(F({p,q}), [&] (const TensorIndex& blockid) {
      std::cout<<__FUNCTION__<<". blockid="<<blockid<<std::endl;
      //auto &blockid = fblock.blockid();
      auto fblock = F.get(blockid);
      auto poff = TCE::offset(blockid[0]);
      auto qoff = TCE::offset(blockid[1]);          
          
      auto fbuf = reinterpret_cast<double*>(fblock.buf());          
      auto bdims = F.block_dims(blockid);
      auto psize = bdims[0].value();
      auto qsize = bdims[1].value();

      auto boffs = TensorVec<size_t>{poff, qoff};
      std::cout<<__FUNCTION__<<". blockid="<<blockid<<std::endl;
      std::cout<<__FUNCTION__<<". block_dims="<<bdims<<std::endl;
      std::cout<<__FUNCTION__<<". block offset="<< boffs <<std::endl;
      for(int p=0, c=0; p<psize; p++) {
        for(int q=0; q<qsize; q++, c++) {
          if(poff+p == qoff+q) {
            //assert(pos < TCE::noab().value()+TCE::nvab().value());
            fdiag[pos++] = fbuf[c];
          }
        }
      }
    });
}

void noga_main(Tensor& D, Tensor& F, Tensor& hT, Tensor& bT, double bdiagsum) {
  using Type = ElementType;
  using Distribution = Tensor::Distribution;

  TensorVec<SymmGroup> indices_ov{SymmGroup{DimType::o}, SymmGroup{DimType::v}};
  TensorVec<SymmGroup> indices_vn{SymmGroup{DimType::v}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> indices_on{SymmGroup{DimType::o}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> indices_oo{SymmGroup{DimType::o}, SymmGroup{DimType::o}};
  TensorVec<SymmGroup> indices_vv{SymmGroup{DimType::v}, SymmGroup{DimType::v}};
  TensorVec<SymmGroup> indices_nn{SymmGroup{DimType::n}, SymmGroup{DimType::n}};
  TensorVec<SymmGroup> t_scalar{};

  Tensor R1{indices_ov, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor R2{indices_oo, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor T{indices_ov, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor Z{indices_oo, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};

  Tensor X_OO{indices_oo, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor X_OV{indices_ov, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor X_VV{indices_vv, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};

  Tensor tmp1{indices_ov, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor tmp2{indices_vv, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor tmp3{indices_vv, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor tmp0{indices_ov, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor tmp4{indices_oo, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor tmp5{indices_oo, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};

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

  T.init(0.0);
  D.init(0.0);

  double fdiag[TCE::total_dim_size()];
  extract_diag(F, fdiag);

  auto delta = [] (int i, int j) {
    return (i==j) ? 1 : 0;
  };
  
  for (int l1 = 0; l1 < 20; l1++) { //OUTERMOST LOOP - LOOP1

    for (int l2 = 0; l2 < 20; l2++) { /// LOOP2
      R1.init(0.0);

      R1({i, a}) += 1.0 * F({i, a});
      R1({i, a}) += -1.0 * (F({i, b}) * X_VV({b, a}));
      R1({i, a}) += -1.0 * (F({i, m}) * X_OV({m, a}));
      R1({i, a}) += X_OO({i, j}) * F({j, a});

      tmp0.init(0.0);
      tmp0({i, b}) += -1.0 * (X_OO({i, j}) * F({j, b}));
      R1({i, a}) += 1.0 * tmp0({i, b}) * X_VV({b, a});
 
      tmp1.init(0.0);
      tmp1({j, a}) += 1.0 * F({j, m}) * X_OV({m, a});
      R1({i, a}) += -1.0 * (X_OO({i, j}) * tmp1({j, a}));
      R1({i, a}) += 1.0 * (X_OV({i, b}) * F({b, a}));
      
      tmp2.init(0.0);
      tmp2({b, a}) += 1.0 * (F({b, c}) * X_VV({c, a}));
      R1({i, a}) += -1.0 * (X_OV({i, b}) * tmp2({b, a}));
      
      tmp3.init(0.0);
      tmp3({b, a}) += 1.0 * (F({b, m}) * X_OV({m, a}));
      R1({i, a}) += -1.0 * (X_OV({i, b}) * tmp3({b, a}));

      /// @todo Denominator can be a scalar for now
      //T({i,a}) += R1({i,a}) / (F({a,a}) - F({i,i}));
      tensor_map(T({i,a}), [&] (Block& tblock) {
          auto &blockid = tblock.blockid();          
          auto ioff = TCE::offset(blockid[0]);
          auto aoff = TCE::offset(blockid[1]);          
          auto r1block = R1.get(blockid);
          
          auto tbuf = reinterpret_cast<double*>(tblock.buf());
          auto r1buf = reinterpret_cast<double*>(r1block.buf());
          
          tblock().init(0.0);
          auto bdims = T.block_dims(blockid);
          auto isize = bdims[0].value();
          auto asize = bdims[1].value();
          for(int i=0, c=0; i<isize; i++) {
            for(int a=0; a<asize; a++, c++) {
              tbuf[c] += r1buf[c] / (fdiag[a] - fdiag[i]);
            }
          }
        });
    }
    
    Z.init(0.0);
    Z({i, j}) += -1.0 * (T({i, e}) * T({j, e}));

    // following loop solves this equation:
    // r_ij = D_ij - delta_ij - D_im * Z_mj
    // D_ij = R_ij / (delta_ij + Z_ij)
    /// NOTE: delta({i,j}) is a Unit matrix: 0 for i!=j 1 for i=j

    for (int l3 = 0; l3 < 10; l3++) {  // LOOP 3
      R2.init(0.0);
      tmp4({i, j}) += 1.0 * (D({i, m}) * Z({m, j}));
      tmp5.init(0.0);
      /// tmp5({i,j}) += delta({i, j}) - tmp4({i, j});
      tensor_map(tmp5({i,j}), [&] (Block& t5block) {
          auto &blockid = t5block.blockid();
          auto ioff = TCE::offset(blockid[0]);
          auto joff = TCE::offset(blockid[1]);
          auto t5buf = reinterpret_cast<double*>(t5block.buf());          
          t5block().init(0.0);
          auto bdims = tmp5.block_dims(blockid);
          auto isize = bdims[0].value();
          auto jsize = bdims[1].value();
          for(int i=0, c=0; i<isize; i++) {
            for(int j=0; j<jsize; j++, c++) {
              t5buf[c] += delta(ioff+i, joff+j);
            }
          }
        });
      tmp5({i,j}) += -1.0 * tmp4({i,j});
      R2({i,j}) += D({i,j});
      R2({i,j}) += -1.0 * tmp5({i,j});
      /// D({i, j}) += R2({i, j}) / (delta({i, j}) + Z({i, j}));
      tensor_map(D({i,j}), [&] (Block& dblock) {
          auto &blockid = dblock.blockid();
          
          auto ioff = TCE::offset(blockid[0]);
          auto joff = TCE::offset(blockid[1]);
          
          auto r2block = R2.get(blockid);
          auto zblock = Z.get(blockid);
          
          auto r2buf = reinterpret_cast<double*>(r2block.buf());
          auto zbuf = reinterpret_cast<double*>(zblock.buf());
          auto dbuf = reinterpret_cast<double*>(dblock.buf());
          
          dblock().init(0.0);
          auto bdims = D.block_dims(blockid);
          auto isize = bdims[0].value();
          auto jsize = bdims[1].value();
          for(int i=0, c=0; i<isize; i++) {
            for(int j=0; j<jsize; j++, c++) {
              dbuf[c] += r2buf[c] / (delta(ioff+i, joff+j) + zbuf[c]);
            }
          }
        });
    }

    X_OV.init(0.0);
    X_OO.init(0.0);
    X_VV.init(0.0);
    X_OV({i, a}) += 1.0 * (D({i, m}) * T({m, a}));
    X_VV({a, b}) += 1.0 * (X_OV({m, a}) * T({m, b}));
    X_OO({i, j}) += -1.0 * (T({i, e}) * X_OV({j, e}));

    noga_fock_build(F, X_OV, X_VV, X_OO, hT, bT, bdiagsum);
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
  using Type = ElementType;
  using Distribution = Tensor::Distribution;

  TensorVec<SymmGroup> indices_oo{SymmGroup{DimType::o}, SymmGroup{DimType::o}};
  TensorVec<SymmGroup> indices_nn{SymmGroup{DimType::n}, SymmGroup{DimType::n}};

  /// @todo what is the initial guess
  Tensor D{indices_oo, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  /// @todo what is the initial guess
  Tensor F{indices_nn, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  ///@todo comes from integrals
  Tensor bT{indices_nn, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};

  ///@todo Who produces this?
  Tensor hT{indices_nn, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};

  hT.allocate();
  D.allocate();
  F.allocate();
  bT.allocate();

  tensor_map(hT(), [](Block &block) {
    int n = 0;
    std::generate_n(reinterpret_cast<double *>(block.buf()), block.size(), [&]() { return n++; });
  });

  tensor_map(bT(), [](Block &block) {
      int n = 0;
      std::generate_n(reinterpret_cast<double *>(block.buf()), block.size(), [&]() { return n++; });
    });

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
  using Type = ElementType;
  using Distribution = Tensor::Distribution;

  TensorVec<SymmGroup> indices_oo{SymmGroup{DimType::o}, SymmGroup{DimType::o}};
  TensorVec<SymmGroup> indices_o{SymmGroup{DimType::o}};
  TensorVec<SymmGroup> indices_nn{SymmGroup{DimType::n}, SymmGroup{DimType::n}};

  Tensor D{indices_oo, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor F{indices_nn, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor bT{indices_nn, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  Tensor hT{indices_nn, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};

  hT.allocate();
  D.allocate();
  F.allocate();
  bT.allocate();

  Tensor d1{indices_o, ElementType::double_precision, Distribution::tce_nwma, 1, irrep_t, false};
  d1.allocate();

  auto oplist = OpList()
      .op<2,0>(hT(), [] (auto i, auto j, auto &ival) {
          ival = i+j;
        })
      .op<1,0>(d1(), [] (auto i, auto &ival) { ival = 0; })
      .op<2,0>(bT(), [] (auto i, auto j, auto &ival) {
          ival = i-j;
        });
      
  {
    auto fn = [] (auto i, auto &lval) {
      lval = i;
    };
    MapOp<decltype(fn),1,0>{d1(), fn}.execute();
  }
  
  {
    int n=0;
    auto fn = [&] (auto i, auto j, auto &lval) {
      lval = n++;
    };
    auto op = MapOp<decltype(fn),2,0>{hT(), fn};
  }

  auto op = mapop_create<2,0>(hT(), [] (auto i, auto j, auto &ival) {
      ival = i+j;
    });

  // auto op = scanop_create<2,0>(hT(), [] (auto i, auto j, auto &ival) {
  //     Expects(std::abs(ival) < 1.0e-12);
  //   });

  // tensor_map(hT(), [](Block &block) {
  //   int n = 0;
  //   std::generate_n(reinterpret_cast<double *>(block.buf()), block.size(), [&]() { return n++; });
  // });

  // tensor_map(bT(), [](Block &block) {
  //     int n = 0;
  //     std::generate_n(reinterpret_cast<double *>(block.buf()), block.size(), [&]() { return n++; });
  //   });

  // double bdiagsum;
  // {
  //   std::vector<double> bdiag(TCE::total_dim_size());
  //   extract_diag(bT, &bdiag[0]);
  //   bdiagsum = 0;
  //   for(auto &b: bdiag) {
  //     bdiagsum += b;
  //   }
  // }
  
  d1.destruct();
  hT.destruct();
  bT.destruct();
  D.destruct();
  F.destruct();
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
