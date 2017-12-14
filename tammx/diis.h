#ifndef TAMMX_DIIS_H_
#define TAMMX_DIIS_H_

#include "tammx/tammx.h"
#include <Eigen/Dense>
#include "ga.h"

namespace tammx {

template<typename T>
inline void
jacobi(const ProcGroup& pg,
       Tensor<T>& d_r, Tensor<T>& d_t, T shift, bool transpose, T* p_evl_sorted) {
  EXPECTS(transpose == false);
  // std::cout << "shift=" << shift << std::endl;
  block_parfor(pg, d_r(), [&] (const BlockDimVec& blockid) {
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
  //GA_Sync();
}


template<typename T>
inline void
jacobi(ExecutionContext& ec,
       Tensor<T>& d_r, Tensor<T>& d_t, T shift, bool transpose, T* p_evl_sorted) {
  EXPECTS(transpose == false);
  // std::cout << "shift=" << shift << std::endl;
  block_parfor(ec.pg(), d_r(), [&] (const BlockDimVec& blockid) {
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
  //GA_Sync();
}

/**
 * @brief dot product between data held in two labeled tensors. Corresponding elements are multiplied.
 *
 * This routine ignores permutation symmetry, and associated symmetrizatin factors
 *
 * @tparam T Type of elements in both tensors
 * @param ec Execution context in which this function is invoked
 * @param lta Labeled tensor A
 * @param ltb labeled Tensor B
 * @return dot product A . B
 */
template<typename T>
inline T
ddot(ExecutionContext& ec, LabeledTensor<T> lta, LabeledTensor<T> ltb) {
  T ret = 0;
  block_for(ec.pg(), lta, [&] (const BlockDimVec& blockid) {
      auto ablock = lta.tensor_->get(blockid);
      auto bblock = ltb.tensor_->get(blockid);
      auto abuf = ablock.buf();
      auto bbuf = bblock.buf();
      size_t sz = ablock.size();
      for(size_t i = 0; i < sz; i++) {
        ret += abuf[i] * bbuf[i];
      }
    });
  return ret;
}

/**
 * @brief DIIS routine
 * @tparam T Type of element in each tensor
 * @param ec Execution context in which this function invoked
 * @param[in] d_rs Vector of R tensors
 * @param[in] d_ts Vector of T tensors
 * @param[out] d_t Vector of T tensors produced by DIIS
 * @pre d_rs.size() == d_ts.size()
 * @pre 0<=i<d_rs.size(): d_rs[i].size() == d_t.size()
 * @pre 0<=i<d_ts.size(): d_ts[i].size() == d_t.size()
 */
template<typename T>
inline void
diis(ExecutionContext& ec,
     std::vector<std::vector<Tensor<T>*>*>& d_rs,
     std::vector<std::vector<Tensor<T>*>*>& d_ts,
     std::vector<Tensor<T>*> d_t) {
  EXPECTS(d_t.size() == d_rs.size());
  int ntensors = d_t.size();
  EXPECTS(ntensors > 0);
  int ndiis = d_rs[0]->size();
  EXPECTS(ndiis > 0);
  for(int i=0; i<ntensors; i++) {
    EXPECTS(d_rs[i]->size() == ndiis);
  }

  using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vector = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Matrix A = Matrix::Zero(ndiis + 1, ndiis + 1);
  Vector b = Vector::Zero(ndiis + 1, 1);
  for(int k=0; k<ntensors; k++) {
    for(int i=0; i<ndiis; i++) {
      for(int j=i; j<ndiis; j++) {
        A(i, j) += ddot(ec, (*d_rs[k]->at(i))(), (*d_rs[k]->at(j))());
      }
    }
  }

  for(int i=0; i<ndiis; i++) {
    for(int j=i; j<ndiis; j++) {
      A(j, i) = A(i, j);
    }
  }
  for(int i=0; i<ndiis; i++) {
    A(i, ndiis) = -1.0;
    A(ndiis, i) = -1.0;
  }

  b(ndiis, 0) = -1;

  // Solve AX = B
  // call dgesv(diis+1,1,a,maxdiis+1,iwork,b,maxdiis+1,info)
  //Vector x = A.colPivHouseholderQr().solve(b);
  Vector x = A.lu().solve(b);

  auto sch = ec.scheduler();
  for(int k=0; k<ntensors; k++) {
    auto &dt = *d_t[k];
    sch.output(dt)
        (dt() = 0);
    for(int j=0; j<ndiis; j++) {
      auto &tb = *d_ts[k]->at(j);
      sch.io(tb)
          (dt() += x(j, 0) * tb());
    }
  }
  //GA_Sync();
  sch.execute();
  //GA_Sync();
  sch.clear();
}

}  // namespace tammx

#endif // TAMMX_DIIS_H_
