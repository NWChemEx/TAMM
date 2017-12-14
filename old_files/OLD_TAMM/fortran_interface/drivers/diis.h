#ifndef TAMMX_DIIS_H_
#define TAMMX_DIIS_H_

#include "tammx/tammx.h"

namespace tammx {

#define MAXDIIS 1000

extern "C" {
/* LU decomposition with partial pivoting */

void dgesv(int* n, int* nrhs, double* a, int* lda, int* ipiv,
                 double* b, int* ldb, int* info);
}

//@todo What should be do wigh transpose
inline void
jacobi(Tensor& d_r, Tensor& d_t, double shift, bool transpose, double *p_evl_sorted) {
  block_for(d_r(), [&] (const BlockDimVec& blockid) {
      auto rblock = d_r.get(blockid);
      auto tblock = d_t.alloc(blockid);
      auto bdims = rblock.block_dims();

      if(d_r.rank() == 2) {
        auto ioff = TCE::offset(blockid[0]);
        auto joff = TCE::offset(blockid[1]);
        auto isize = bdims[0].value();
        auto jsize = bdims[1].value();
        double *rbuf = reinterpret_cast<double*>(rblock.buf());
        double *tbuf = reinterpret_cast<double*>(tblock.buf());
        for(int i=0, c=0; i<isize; i++) {
          for(int j=0; j<jsize; j++, c++) {
            tbuf[c] = rbuf[c] / (-p_evl_sorted[ioff+i] + p_evl_sorted[joff+j] + shift);
          }
        }
        d_t.add(tblock);
      } else if(d_r.rank() == 4) {
        assert(0);  // $todo implement
      }
      else {
        assert(0);  // @todo implement
      }
    });
}

class DIIS {
 public:
  DIIS(Tensor::Distribution distribution, bool transpose, double zshiftl, int ndiis, int ntensors, double *p_evl_sorted)
      : distribution_{distribution},
        transpose_{transpose},
        zshiftl_{zshiftl},
        ndiis_{ndiis},
        allocated_{false},
        iter_{0},
        ntensors_{ntensors},
        d_rs_(ntensors),
        d_ts_(ntensors),
        p_evl_sorted_{p_evl_sorted} {}

  void next(const std::vector<Tensor*>& d_r,
            const std::vector<Tensor*>& d_t) {
    Expects(d_r.size() == d_t.size());
    Expects(d_r.size() == ntensors_);
    Expects(iter_ == 0 || allocated_);

    if (iter_ == 0 && !allocated_) {
      for (int i = 0; i < ntensors_; i++) {
        auto indices = d_r[i]->indices();
        auto eltype = d_r[i]->element_type();
        auto nupper = d_r[i]->nupper_indices();
        auto irrep = d_r[i]->irrep();
        auto spin_restricted = d_r[i]->spin_restricted();
        for (int j = 0; j < ndiis_; j++) {
          d_rs_[i].push_back(new Tensor{indices, eltype, distribution_,
              nupper, irrep, spin_restricted});
          d_ts_[j].push_back(new Tensor{indices, eltype, distribution_,
              nupper, irrep, spin_restricted});
          d_rs_[i].back()->allocate();
          d_ts_[i].back()->allocate();
          d_rs_[i].back()->init(0);
          d_ts_[i].back()->init(0);
        }
      }
      allocated_ = true;
    }

    if (iter_ < ndiis_-1) {
      for (int i = 0; i < ntensors_; i++) {
        double shift = -1.0 * d_rs_[i].back()->rank()/2 * zshiftl_;
        jacobi(*d_r[i], *d_t[i], shift, transpose_, p_evl_sorted_);
      }
      return;
    }
    for (int i = 0; i < ntensors_; i++) {
      (*d_rs_[i][iter_])() += (*d_r[i])();
      (*d_ts_[i][iter_])() += (*d_t[i])();
    }
    iter_ += 1;
    // static allocation changed to dynamic
    // double a[ndiis_+1][ndiis_+1];
    double a[ndiis_+1][ndiis_+1];
    for (int i = 0; i < ndiis_; i++) {
      for (int j = 0; j < ndiis_; j++) {
        a[i][j] = 0;
        for (int k = 0; k < ntensors_; k++) {
          a[i][j] += ddot(*d_rs_[k][i], *d_rs_[k][j]);
        }
      }
    }
    for (int i = 0; i < ndiis_; i++) {
      a[i][ndiis_] = -1.0;
      a[ndiis_][i] = -1.0;
    }
    a[ndiis_][ndiis_] = 0;

    double b[ndiis_+1];
    // static allocation changed to dynamic
    // double b[ndiis_+1];
    std::fill_n(b, b+ndiis_, 0);
    b[ndiis_] = -1;

    /* Solve AX = B */
    int iwork[ndiis_+1];
    int maxdiis = MAXDIIS;
    int n_dgesv = ndiis_ + 1;
    int lda_dgesv = maxdiis+1;
    int ldb_dgesv = maxdiis+1;
    int nrhs = 1;
    int info;
    // dgesv(&n_dgesv, &nrhs, &a[0][0], &lda_dgesv, &iwork[0], &b[0],
    //         &ldb_dgesv, &info);
    //if (info > 0) nodezero_print("tce_diis: LU decompositon failed \n " + info);

    for (int i = 0; i < ntensors_; i++) {
      d_t[i]->init(0);
      for (int j = 0; j < ndiis_; j++) {
        (*d_t[i])() += b[j] * (*d_ts_[i][j])();
      }
    }
    iter_ = 0;

  }



  void destruct() {
    for (auto &pdrvec : d_rs_) {
      for (auto &pdr : pdrvec) {
        pdr->destruct();
        delete pdr;
      }
    }
    for (auto &pdtvec : d_ts_) {
      for (auto &pdt : pdtvec) {
        pdt->destruct();
        delete pdt;
      }
    }
    d_rs_.clear();
    d_ts_.clear();
    iter_ = 0;
    allocated_ = false;
  }

  ~DIIS() {
    Expects(!allocated_);
  }

 private:
  double ddot(Tensor& d_a, Tensor& d_b) {
    Tensor dot{TensorVec<SymmGroup>{}, Tensor::Type::double_precision,
        distribution_, 0, Irrep{0}, false};
    dot.init(0);
    dot() += d_a() * d_b();
    auto block = dot.get({});
    return *reinterpret_cast<double*>(block.buf());
  }

  Tensor::Distribution distribution_;
  bool allocated_;
  bool transpose_;
  double zshiftl_;
  int ndiis_, iter_, ntensors_;
  std::vector<std::vector<Tensor*>> d_rs_, d_ts_;
  double *p_evl_sorted_;
};  // class DIIS

}  // namespace tammx

#endif // TAMMX_DIIS_H_
