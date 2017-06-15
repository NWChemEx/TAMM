#ifndef TAMMX_DIIS_H_
#define TAMMX_DIIS_H_

#include "tammx/tammx.h"

namespace tammx {


//@todo What should be do with transpose
template<typename T>
inline void
jacobi(Tensor<T>& d_r, Tensor<T>& d_t, T shift, bool transpose, T* p_evl_sorted) {
  Expects(transpose == false);
  block_for(d_r(), [&] (const TensorIndex& blockid) {
      auto rblock = d_r.get(blockid);
      auto tblock = d_t.alloc(blockid);
      auto bdims = rblock.block_dims();
      
      if(d_r.rank() == 2) {
        auto ioff = TCE::offset(blockid[0]);
        auto joff = TCE::offset(blockid[1]);
        auto isize = bdims[0].value();
        auto jsize = bdims[1].value();
        T* rbuf = rblock.buf();
        T* tbuf = tblock.buf();
        for(int i=0, c=0; i<isize; i++) {
          for(int j=0; j<jsize; j++, c++) {
            tbuf[c] = rbuf[c] / (-p_evl_sorted[ioff+i] + p_evl_sorted[joff+j] + shift);
          }
        }
        d_t.add(tblock.blockid(), tblock);
      } else if(d_r.rank() == 4) {
        auto off = rblock.block_offset();
        TensorVec<int64_t> ioff;
        for(auto x: off) {
          ioff.push_back(x.value());
        }
        TensorVec<int64_t> isize;
        for(auto x: bdims) {
          isize.push_back(x.value());
        }
        T* rbuf = rblock.buf();
        T* tbuf = tblock.buf();
        for(int i0=0, c=0; i0<isize[0]; i0++) {
          for(int i1=0; i1<isize[1]; i1++) {
            for(int i2=0; i2<isize[2]; i2++) {
              for(int i3=0; i3<isize[3]; i3++, c++) {
                tbuf[c] = rbuf[c] / (- p_evl_sorted[ioff[0]+i0] - p_evl_sorted[ioff[1]+i1]
                                     + p_evl_sorted[ioff[2]+i2] + p_evl_sorted[ioff[3]+i3]
                                     + shift);
              }
            }
          }
        }
        d_t.add(tblock.blockid(), tblock);
      }
      else {
        assert(0);  // @todo implement
      }
    });
}

// @todo @bug @fixme This code has not been checked. 
template<typename T>
inline void
diis(Scheduler& sch,
     std::vector<std::vector<Tensor<T>*>*>& d_rs,
     std::vector<Tensor<T>*> d_t) {
  Expects(d_t.size() == d_rs.size());
  int ndiis = d_t.size();
  Expects(ndiis > 0);
  int ntensors = d_rs[0]->size();
  Expects(ntensors > 0);
  for(int i=0; i<ndiis; i++) {
    Expects(d_rs[i]->size() == ntensors);
  }
  
  //T aexp[ndiis+1][ndiis+1][ntensors];
  Scalar<T> aexp[ndiis+1][ndiis+1][ntensors];
  for(int i=0; i<ndiis; i++) {
    for(int j=0; j<ndiis; j++) {
      for(int k=0; k<ntensors; k++) {
        auto &ta = *d_rs[k]->at(i);
        auto &tb = *d_rs[k]->at(j);
        sch.alloc(aexp[i][j][k])
            (aexp[i][j][k]() = ta() * tb());
      }
    }
  }
  sch.execute();
  //sch.clear();
  
  T a[ndiis+1][ndiis+1];
  for(int i=0; i<ndiis; i++) {
    for(int j=0; j<ndiis; j++) {
      a[i][j] = 0;
      for(int k=0; k<ntensors; k++) {
        a[i][j] += aexp[i][j][k].value();
      }
    }
  }
  for(int i=0; i<ndiis; i++) {
    a[i][ndiis] = -1.0;
    a[ndiis][i] = -1.0;
  }
  a[ndiis][ndiis] = 0;
  a[ndiis][ndiis] = 0;
  
  T b[ndiis+1];
  std::fill_n(b, b+ndiis, T{0});
  b[ndiis] = -1;
  
  // Solve AX = B
  // call dgesv(diis+1,1,a,maxdiis+1,iwork,b,maxdiis+1,info)
  
  for(int i=0; i<ntensors; i++) {
    auto &dt = *d_t[i];
    sch(dt() = 0);
    for(int j=0; j<ndiis; j++) {
      auto &tb = *d_rs[i]->at(j);
      sch(dt() += b[j] * tb());
    }
  }
  sch.execute();
  //sch.clear();
}

#if 0
template<typename T>
class DIIS {
 public:
  DIIS(, bool transpose, double zshiftl, int ndiis, int ntensors, double *p_evl_sorted)
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

    if(iter_ == 0 && !allocated_) {
      for(int i=0; i<ntensors_; i++) {
        auto indices = d_r[i]->indices();
        auto eltype = d_r[i]->element_type();
        auto nupper = d_r[i]->nupper_indices();
        auto irrep = d_r[i]->irrep();
        auto spin_restricted = d_r[i]->spin_restricted();
        for(int j=0; j<ndiis_; j++) {
          d_rs_[i].push_back(new Tensor{indices, eltype, distribution_, nupper, irrep, spin_restricted});
          d_ts_[j].push_back(new Tensor{indices, eltype, distribution_, nupper, irrep, spin_restricted});
          d_rs_[i].back()->allocate();
          d_ts_[i].back()->allocate();
          d_rs_[i].back()->init(0);
          d_ts_[i].back()->init(0);
        }      
      }
      allocated_ = true;
    }

    if(iter_ < ndiis_-1) {
      for(int i=0; i<ntensors_; i++) {
        double shift = -1.0 * d_rs_[i].back()->rank()/2 * zshiftl_;
        jacobi(*d_r[i], *d_t[i], shift, transpose_, p_evl_sorted_);
      }
      return;
    }
    for(int i=0; i<ntensors_; i++) {
      (*d_rs_[i][iter_])() += (*d_r[i])();
      (*d_ts_[i][iter_])() += (*d_t[i])();
    }
    iter_ += 1;
    double a[ndiis_+1][ndiis_+1];
    for(int i=0; i<ndiis_; i++) {
      for(int j=0; j<ndiis_; j++) {
        a[i][j] = 0;
        for(int k=0; k<ntensors_; k++) {
          a[i][j] += ddot(*d_rs_[k][i], *d_rs_[k][j]);
        }
      }
    }
    for(int i=0; i<ndiis_; i++) {
      a[i][ndiis_] = -1.0;
      a[ndiis_][i] = -1.0;
    }
    a[ndiis_][ndiis_] = 0;

    double b[ndiis_+1];
    std::fill_n(b, b+ndiis_, 0);
    b[ndiis_] = -1;

    // Solve AX = B
    // call dgesv(diis+1,1,a,maxdiis+1,iwork,b,maxdiis+1,info)

    for(int i=0; i<ntensors_; i++) {
      d_t[i]->init(0);
      for(int j=0; j<ndiis_; j++) {
        (*d_t[i])() += b[j] * (*d_ts_[i][j])();
      }
    }
    iter_ = 0;
  }

  void deallocate() {
    for(auto &pdrvec : d_rs_) {
      for(auto &pdr: pdrvec) {
        pdr->destruct();
        delete pdr;
      }
    }
    for(auto &pdtvec : d_ts_) {
      for(auto &pdt: pdtvec) {
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
    Tensor dot{TensorVec<SymmGroup>{}, Tensor::Type::double_precision, distribution_, 0, Irrep{0}, false};
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
#endif

}  // namespace tammx

#endif // TAMMX_DIIS_H_
