#ifndef TAMMX_DIIS_H_
#define TAMMX_DIIS_H_

#include "tammx/tammx.h"
#include <Eigen/Dense>

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

template<typename T>
inline void
diis(Scheduler& sch,
     std::vector<std::vector<Tensor<T>*>*>& d_rs,
     std::vector<Tensor<T>*> d_t) {
  Expects(d_t.size() == d_rs.size());
  int ntensors = d_t.size();
  Expects(ntensors > 0);
  int ndiis = d_rs[0]->size();
  Expects(ndiis > 0);
  for(int i=0; i<ntensors; i++) {
    Expects(d_rs[i]->size() == ndiis);
  }
  
  Scalar<T> aexp[ntensors][ndiis][ndiis];
  for(int k=0; k<ntensors; k++) {
    for(int i=0; i<ndiis; i++) {
      for(int j=0; j<ndiis; j++) {
        aexp[k][i][j].alloc(sch.pg(), sch.default_distribution(), sch.default_memory_manager());
      }
    }
  }
  for(int k=0; k<ntensors; k++) {
    for(int i=0; i<ndiis; i++) {
      auto &ta = *d_rs[k]->at(i);
      sch.io(ta);
    }
  }
  
  for(int k=0; k<ntensors; k++) {
    for(int i=0; i<ndiis; i++) {
      for(int j=0; j<ndiis; j++) {
        auto &ta = *d_rs[k]->at(i);
        auto &tb = *d_rs[k]->at(j);
        sch.output(aexp[k][i][j])
            (aexp[k][i][j]() = 0)
            (aexp[k][i][j]() += ta() * tb());
      }
    }
  }
  sch.execute();
  sch.clear();
  
  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Matrix A = Matrix::Zero(ndiis + 1, ndiis + 1);
  Vector b = Vector::Zero(ndiis + 1, 1);
  for(int k=0; k<ntensors; k++) {
    for(int i=0; i<ndiis; i++) {
      for(int j=0; j<ndiis; j++) {
        A(i, j) += aexp[k][i][j].value();
      }
    }
  }
  for(int i=0; i<ndiis; i++) {
    A(i, ndiis) = -1.0;
    A(ndiis, i) = -1.0;
  }
  
  b(ndiis, 0) = -1;

  for(int k=0; k<ntensors; k++) {
    for(int i=0; i<ndiis; i++) {
      for(int j=0; j<ndiis; j++) {
        aexp[k][i][j].dealloc();
      }
    }
  }
  // Solve AX = B
  // call dgesv(diis+1,1,a,maxdiis+1,iwork,b,maxdiis+1,info)
  Vector x = A.colPivHouseholderQr().solve(b);
  
  for(int k=0; k<ntensors; k++) {
    auto &dt = *d_t[k];
    sch.output(dt)
        (dt() = 0);
    for(int j=0; j<ndiis; j++) {
      auto &tb = *d_rs[k]->at(j);
      sch.io(tb)
          (dt() += x(j, 0) * tb());
    }
  }
  sch.execute();
  sch.clear();
}

}  // namespace tammx

#endif // TAMMX_DIIS_H_
