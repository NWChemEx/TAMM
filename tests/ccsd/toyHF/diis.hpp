#ifndef TAMM_DIIS_HPP_
#define TAMM_DIIS_HPP_

#include "ga.h"
#include "tamm/tamm.hpp"
#include <Eigen/Dense>

namespace tamm {

template<typename T>
inline void jacobi(ExecutionContext& ec, Tensor<T>& d_r, Tensor<T>& d_t,
                   T shift, bool transpose, std::vector<double>& evl_sorted, const TAMM_SIZE& noab) {
    // EXPECTS(transpose == false);
    block_for(ec.pg(), d_r(), [&](IndexVector blockid) {
        const TAMM_SIZE rsize = d_r.block_size(blockid);
        std::vector<T> rbuf(rsize);
        d_r.get(blockid, rbuf);

        const TAMM_SIZE tsize = d_t.block_size(blockid);

        std::vector<T> tbuf(tsize);

        auto& rtiss      = d_r.tiled_index_spaces();
        auto rblock_dims = d_r.block_dims(blockid);

        std::vector<double> p_evl_sorted_occ(noab);
        std::vector<double> p_evl_sorted_virt(evl_sorted.size()-noab);
        std::copy(evl_sorted.begin(), evl_sorted.begin() + noab,
                  p_evl_sorted_occ.begin());
        std::copy(evl_sorted.begin() + noab, evl_sorted.end(),
                  p_evl_sorted_virt.begin());

        if(d_r.num_modes() == 2) {
            auto ioff  = rtiss[0].tile_offset(blockid[0]);
            auto joff  = rtiss[1].tile_offset(blockid[1]);
            auto isize = rblock_dims[0];
            auto jsize = rblock_dims[1];

            if(!transpose) {
                for(int i = 0, c = 0; i < isize; i++) {
                    for(int j = 0; j < jsize; j++, c++) {
                        tbuf[c] =
                          rbuf[c] / (-p_evl_sorted_virt[ioff + i] +
                                     p_evl_sorted_occ[joff + j] + shift);
                    }
                }
            } else {
                for(int i = 0, c = 0; i < isize; i++) {
                    for(int j = 0; j < jsize; j++, c++) {
                        tbuf[c] =
                          rbuf[c] / (-p_evl_sorted_occ[joff + j] +
                                     p_evl_sorted_virt[ioff + i] + shift);
                    }
                }
            }
            d_t.add(blockid, tbuf);
        } else if(d_r.num_modes() == 4) {
            const int ndim = 4;
            std::array<int, ndim> rblock_offset{
             rtiss[0].tile_offset(blockid[0]),   
             rtiss[1].tile_offset(blockid[1]),
             rtiss[2].tile_offset(blockid[2]),
             rtiss[3].tile_offset(blockid[3])                
            };
            // for(auto i = 0; i < ndim; i++) {
            //     rblock_offset[i] = rtiss[i].tile_offset(blockid[i]);
            // }
            std::vector<size_t> ioff;
            for(auto x : rblock_offset) { ioff.push_back(x); }
            std::vector<size_t> isize;
            for(auto x : rblock_dims) { isize.push_back(x); }

            if(!transpose) {
                for(int i0 = 0, c = 0; i0 < isize[0]; i0++) {
                    for(int i1 = 0; i1 < isize[1]; i1++) {
                        for(int i2 = 0; i2 < isize[2]; i2++) {
                            for(int i3 = 0; i3 < isize[3]; i3++, c++) {
                                tbuf[c] =
                                  rbuf[c] /
                                  (-p_evl_sorted_virt[ioff[0] + i0] -
                                   p_evl_sorted_virt[ioff[1] + i1] +
                                   p_evl_sorted_occ[ioff[2] + i2] +
                                   p_evl_sorted_occ[ioff[3] + i3] + shift);
                            }
                        }
                    }
                }
            } else {
                for(int i0 = 0, c = 0; i0 < isize[0]; i0++) {
                    for(int i1 = 0; i1 < isize[1]; i1++) {
                        for(int i2 = 0; i2 < isize[2]; i2++) {
                            for(int i3 = 0; i3 < isize[3]; i3++, c++) {
                                tbuf[c] =
                                  rbuf[c] /
                                  (-p_evl_sorted_occ[ioff[2] + i2] -
                                   p_evl_sorted_occ[ioff[3] + i3] +
                                   p_evl_sorted_virt[ioff[0] + i0] +
                                   p_evl_sorted_virt[ioff[1] + i1] + shift);
                            }
                        }
                    }
                }
            }
            d_t.add(blockid, tbuf);
        } else {
            assert(0); // @todo implement
        }
    });
    // GA_Sync();
}

/**
 * @brief dot product between data held in two labeled tensors. Corresponding
 * elements are multiplied.
 *
 * This routine ignores permutation symmetry, and associated symmetrizatin
 * factors
 *
 * @tparam T Type of elements in both tensors
 * @param ec Execution context in which this function is invoked
 * @param lta Labeled tensor A
 * @param ltb labeled Tensor B
 * @return dot product A . B
 */
template<typename T>
inline T ddot(ExecutionContext& ec, LabeledTensor<T> lta,
              LabeledTensor<T> ltb) {
    T ret = 0;
    block_for(ec.pg(), lta, [&](IndexVector blockid) {
        Tensor<T> atensor     = lta.tensor();
        const TAMM_SIZE asize = atensor.block_size(blockid);
        std::vector<T> abuf(asize);

        Tensor<T> btensor     = ltb.tensor();
        const TAMM_SIZE bsize = btensor.block_size(blockid);
        std::vector<T> bbuf(bsize);

        const size_t sz = asize;
        for(size_t i = 0; i < sz; i++) { ret += abuf[i] * bbuf[i]; }
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
inline void diis(ExecutionContext& ec,
                 std::vector<std::vector<Tensor<T>*>*>& d_rs,
                 std::vector<std::vector<Tensor<T>*>*>& d_ts,
                 std::vector<Tensor<T>*> d_t) {
                 //,const TiledIndexSpace& MO) {

    // const TiledIndexSpace& O = MO("occ");
    // const TiledIndexSpace& V = MO("virt");
    // Tensor<T> i1{O, V};

    // TiledIndexLabel p1, p2, p3;
    // TiledIndexLabel h1, h2, h3;

    // std::tie(p1, p2, p3) = MO.labels<3>("virt");
    // std::tie(h1, h2, h3) = MO.labels<3>("occ");

    EXPECTS(d_t.size() == d_rs.size());
    int ntensors = d_t.size();
    EXPECTS(ntensors > 0);
    int ndiis = d_rs[0]->size();
    EXPECTS(ndiis > 0);
    for(int i = 0; i < ntensors; i++) { EXPECTS(d_rs[i]->size() == ndiis); }

    using Matrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Vector =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Matrix A = Matrix::Zero(ndiis + 1, ndiis + 1);
    Vector b = Vector::Zero(ndiis + 1, 1);
    for(int k = 0; k < ntensors; k++) {
        for(int i = 0; i < ndiis; i++) {
            for(int j = i; j < ndiis; j++) {
                Tensor<T> d_r1{};
                Tensor<T>::allocate(&ec,d_r1);
                Tensor<T>& t1 = *d_rs[k]->at(i);
                Tensor<T>& t2 = *d_rs[k]->at(j);
                Scheduler{&ec}(d_r1() = 0).execute();
                //A(i, j) += ddot(ec, (*d_rs[k]->at(i))(), (*d_rs[k]->at(j))());
                if (t1.num_modes() == 2){
                    Scheduler{&ec}(d_r1() += t1() * t2()).execute();
                }
                else if(t1.num_modes() == 4){
                    Scheduler{&ec}(d_r1() += 0.25 * t1() * t2()).execute();
                }
                T r1;
                d_r1.get({}, {&r1, 1});
                A(i,j) += r1;
                Tensor<T>::deallocate(d_r1);
            }
        }
    }

    for(int i = 0; i < ndiis; i++) {
        for(int j = i; j < ndiis; j++) { A(j, i) = A(i, j); }
    }
    for(int i = 0; i < ndiis; i++) {
        A(i, ndiis) = -1.0;
        A(ndiis, i) = -1.0;
    }

    b(ndiis, 0) = -1;

    // Solve AX = B
    // call dgesv(diis+1,1,a,maxdiis+1,iwork,b,maxdiis+1,info)
    // Vector x = A.colPivHouseholderQr().solve(b);
    Vector x = A.lu().solve(b);

    auto sch = Scheduler{&ec};
    for(int k = 0; k < ntensors; k++) {
        Tensor<T>& dt = *d_t[k];
        sch(dt() = 0);
        for(int j = 0; j < ndiis; j++) {
            auto& tb = *d_ts[k]->at(j);
            sch(dt() += x(j, 0) * tb());
        }
    }
    // GA_Sync();
    sch.execute();
    // GA_Sync();
}

} // namespace tamm

#endif // TAMM_DIIS_HPP_
