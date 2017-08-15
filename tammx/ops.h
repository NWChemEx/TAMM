#ifndef TAMMX_OPS_H_
#define TAMMX_OPS_H_

#include "tammx/tensor.h"
#include "tammx/labeled-block.h"
#include "tammx/labeled-tensor.h"
#include "tammx/util.h"

namespace tammx {

/////////////////////////////////////////////////////////////////////
//         operators
/////////////////////////////////////////////////////////////////////

class Op {
 public:
  virtual void execute() = 0;
  virtual ~Op() {}
};


template<typename T, typename LabeledTensorType>
struct SetOp : public Op {
  void execute() override;

  SetOp(T value, LabeledTensorType& lhs, ResultMode mode)
      : value_{value},
        lhs_{lhs},
        mode_{mode} {}

  T value_;
  LabeledTensorType lhs_;
  ResultMode mode_;
};

template<typename T, typename LabeledTensorType>
struct AddOp : public Op {
  void execute() override;

  AddOp(T alpha, const LabeledTensorType& lhs, const LabeledTensorType& rhs, ResultMode mode)
      : alpha_{alpha},
        lhs_{lhs},
        rhs_{rhs},
        mode_{mode} { }

  T alpha_;
  LabeledTensorType lhs_, rhs_;
  ResultMode mode_;
};

template<typename T, typename LabeledTensorType>
struct MultOp : public Op {
  void execute() override;

  MultOp(T alpha, const LabeledTensorType& lhs, const LabeledTensorType& rhs1,
         const LabeledTensorType& rhs2, ResultMode mode)
      : alpha_{alpha},
        lhs_{lhs},
        rhs1_{rhs1},
        rhs2_{rhs2},
        mode_{mode} { }

  T alpha_;
  LabeledTensorType lhs_, rhs1_, rhs2_;
  ResultMode mode_;
};

template<typename TensorType>
struct AllocOp: public Op {
  void execute() override {
    tensor_->alloc(pg, distribution_, memory_manager_);
  }

  AllocOp(TensorType& tensor, ProcGroup pg, Distribution* distribution, MemoryManager* memory_manager)
      : tensor_{&tensor},
        distribution_{distribution},
        memory_manager_{memory_manager} {}

  TensorType *tensor_;
  ProcGroup pg;
  Distribution* distribution_;
  MemoryManager* memory_manager_;
};

template<typename TensorType>
struct DeallocOp: public Op {
  void execute() override {
    tensor_->dealloc();
  }

  DeallocOp(TensorType* tensor)
      : tensor_{tensor} {}

  TensorType *tensor_;
};


template<typename LabeledTensorType, typename Func, int N>
struct MapOp : public Op {
  using RHS = std::array<LabeledTensorType, N>;
  using T = typename LabeledTensorType::element_type;
  using RHS_Blocks = std::array<Block<T>, N>;

  void execute() override {
    auto &lhs_tensor = *lhs_.tensor_;
    auto lambda = [&] (const TensorIndex& blockid) {
      auto size = lhs_tensor.block_size(blockid);
      if(!(lhs_tensor.nonzero(blockid) && lhs_tensor.spin_unique(blockid) && size > 0)) {
        return;
      }
      auto lblock = lhs_tensor.alloc(blockid);
      auto rhs_blocks = get_blocks(rhs_, blockid);
      std::array<T*,N> rhs_bufs;
      for(int i=0; i<N; i++) {
        rhs_bufs[i] = rhs_blocks[i].buf();
      }
      for(int i=0; i<size; i++) {
        func_(lblock.buf(), rhs_bufs, i);
      }
      if(mode_ == ResultMode::update) {
        lhs_tensor.add(lblock);
      } else if (mode_ == ResultMode::set) {
        lhs_tensor.put(lblock);
      } else {
        assert(0);
      }
    };
    auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  MapOp(LabeledTensorType& lhs, Func func, RHS& rhs, ResultMode mode = ResultMode::set)
      : lhs_{lhs},
        func_{func},
        rhs_{rhs},
        mode_{mode} {}

  RHS_Blocks get_blocks(RHS& rhs, const TensorIndex& id) {
    RHS_Blocks blocks;
    for(int i=0; i<rhs.size(); i++) {
      blocks[i] = rhs[i].get(id);
    }
    return blocks;
  }
  
  LabeledTensorType& lhs_;
  Func func_;
  std::array<LabeledTensorType, N> rhs_;
  ResultMode mode_;
};

template<typename LabeledTensorType, typename Func, int N>
struct MapIdOp : public Op {
  using RHS = std::array<LabeledTensorType, N>;
  using T = typename LabeledTensorType::element_type;
  using RHS_Blocks = std::array<Block<T>, N>;

  void execute() override {
    auto &lhs_tensor = *lhs_.tensor_;
    auto lambda = [&] (const TensorIndex& blockid) {
      auto size = lhs_tensor.block_size(blockid);
      if(!(lhs_tensor.nonzero(blockid) && lhs_tensor.spin_unique(blockid) && size > 0)) {
        return;
      }
      auto lblock = lhs_tensor.alloc(blockid);
      auto rhs_blocks = get_blocks(rhs_, blockid);
      std::array<T*,N> rhs_bufs;
      for(int i=0; i<N; i++) {
        rhs_bufs[i] = rhs_blocks[i].buf();
      }

      auto lo = lhs_tensor.block_offset(blockid);
      auto dims = lhs_tensor.block_dims(blockid);
      auto hi = lo;
      for(int i=0; i<hi.size(); i++) {
        hi[i] += dims[i];
      }
      auto itr = lo;
      
      if(lo.size()==0) {
        func_(itr);
      } else if(lo.size() == 1) {
        int i=0;
        for(itr[0]=lo[0]; itr[0]<hi[0]; itr[0]++, i++) {
          func_(lblock.buf(), rhs_bufs, i, itr);
        }
      } else if(lo.size() == 2) {
        int i=0;
        for(itr[0]=lo[0]; itr[0]<hi[0]; itr[0]++) {
          for(itr[1]=lo[1]; itr[1]<hi[1]; itr[1]++, i++) {
            func_(lblock.buf(), rhs_bufs, i, itr);
          }
        }
      } else if(lo.size() == 3) {
        int i=0;
        for(itr[0]=lo[0]; itr[0]<hi[0]; itr[0]++) {
          for(itr[1]=lo[1]; itr[1]<hi[1]; itr[1]++) {
            for(itr[2]=lo[2]; itr[2]<hi[2]; itr[2]++, i++) {
              func_(lblock.buf(), rhs_bufs, i, itr);
            }
          }
        }
      } else if(lo.size() == 4) {
        int i=0;
        for(itr[0]=lo[0]; itr[0]<hi[0]; itr[0]++) {
          for(itr[1]=lo[1]; itr[1]<hi[1]; itr[1]++) {
            for(itr[2]=lo[2]; itr[2]<hi[2]; itr[2]++) {
              for(itr[3]=lo[3]; itr[3]<hi[3]; itr[3]++, i++) {
                func_(lblock.buf(), rhs_bufs, i, itr);
              }
            }
          }
        }
      } else {
        // @todo implement
        assert(0);
      }  

      // for(int i=0; i<size; i++) {
      //   func_(lblock.buf(), rhs_bufs, i);
      // }
      if(mode_ == ResultMode::update) {
        lhs_tensor.add(lblock);
      } else if (mode_ == ResultMode::set) {
        lhs_tensor.put(lblock);
      } else {
        assert(0);
      }
    };
    auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  MapIdOp(LabeledTensorType& lhs, Func func, RHS& rhs, ResultMode mode = ResultMode::set)
      : lhs_{lhs},
        func_{func},
        rhs_{rhs},
        mode_{mode} {}

  RHS_Blocks get_blocks(RHS& rhs, const TensorIndex& id) {
    RHS_Blocks blocks;
    for(int i=0; i<rhs.size(); i++) {
      blocks[i] = rhs[i].get(id);
    }
    return blocks;
  }
  
  LabeledTensorType& lhs_;
  Func func_;
  std::array<LabeledTensorType, N> rhs_;
  ResultMode mode_;
};


// #if 0

// template<typename Func, typename LabeledTensorType, unsigned ndim, unsigned nrhs>
// struct MapOp : public Op {
//   void execute();
// };

// /**
//  * @note Works with arbitrary dimensions
//  */
// template<typename Func, typename LabeledTensorType>
// struct MapOp<Func, LabeledTensorType, 0, 0> : public Op {
//   void execute() {
//     // std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
//     auto& lhs_tensor = *lhs_.tensor_;
//     auto lambda = [&] (const TensorIndex& blockid) {
//       auto size = lhs_tensor.block_size(blockid);
//       if(!(lhs_tensor.nonzero(blockid) && lhs_tensor.spin_unique(blockid) && size > 0)) {
//         return;
//       }
//       // std::cerr<<"MapOp. size="<<size<<std::endl;
//       auto lblock = lhs_tensor.alloc(blockid);
//       for(int i=0; i<size; i++) {
//         func_(lblock.buf()[i]);
//       }
//       if(mode_ == ResultMode::update) {
//         lhs_tensor.add(lblock);
//       } else if (mode_ == ResultMode::set) {
//         lhs_tensor.put(lblock);
//       } else {
//         assert(0);
//       }
//     };
//     auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
//     parallel_work(itr_first, itr_first.get_end(), lambda);
//   }

//   MapOp(const LabeledTensorType& lhs, Func func, ResultMode mode = ResultMode::set)
//       : lhs_{lhs},
//         func_{func},
//         mode_{mode} {
//     Expects(lhs_.tensor_ != nullptr);
//   }

//   LabeledTensorType lhs_;
//   Func func_;
//   ResultMode mode_;
// };

// /**
//  * @todo more generic ndimg and rhs versions
//  *
//  */
// template<typename Func, typename LabeledTensorType>
// struct MapOp<Func, LabeledTensorType, 1, 0> : public Op {
//   void execute() {
//     // std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
//     auto& lhs_tensor = *lhs_.tensor_;

//     auto lambda = [&] (const TensorIndex& blockid) {
//       auto size = lhs_tensor.block_size(blockid);
//       if(!(lhs_tensor.nonzero(blockid) && lhs_tensor.spin_unique(blockid) && size > 0)) {
//         return;
//       }
//       auto offset = TCE::offset(blockid[0]);
//       auto lblock = lhs_tensor.alloc(blockid);
//       for(int i=0; i<size; i++) {
//         func_(offset + i, lblock.buf()[i]);
//       }
//       if(mode_ == ResultMode::update) {
//         lhs_tensor.add(lblock);
//       } else if (mode_ == ResultMode::set) {
//         lhs_tensor.put(lblock);
//       } else {
//         assert(0);
//       }
//     };
//     auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
//     parallel_work(itr_first, itr_first.get_end(), lambda);
//   }

//   MapOp(const LabeledTensorType& lhs, Func func, ResultMode mode = ResultMode::set)
//       : lhs_{lhs}, func_{func}, mode_{mode} {
//     Expects(lhs_.tensor_ != nullptr);
//     Expects(lhs_.tensor_->rank() == 1);
//   }

//   LabeledTensorType lhs_;
//   Func func_;
//   ResultMode mode_;
// };

// template<typename Func, typename LabeledTensorType>
// struct MapOp<Func, LabeledTensorType, 2, 0> : public Op {
//   void execute() {
//     std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
//     auto& lhs_tensor = *lhs_.tensor_;

//     auto lambda = [&] (const TensorIndex& blockid) {
//       auto size = lhs_tensor.block_size(blockid);
//       if(!(lhs_tensor.nonzero(blockid) && lhs_tensor.spin_unique(blockid) && size > 0)) {
//         return;
//       }
//       auto bdims = lhs_tensor.block_dims(blockid);
//       auto isize = bdims[0].value();
//       auto jsize = bdims[1].value();
//       auto ioffset = TCE::offset(blockid[0]);
//       auto joffset = TCE::offset(blockid[1]);
//       if(lhs_tensor.nonzero(blockid) && isize*jsize > 0) {
//         auto lblock = lhs_tensor.alloc(blockid);
//         for(int i=0, c=0; i<isize; i++) {
//           for(int j=0; j<jsize; j++, c++) {
//             func_(ioffset+i, joffset+j, lblock.buf()[c]);
//           }
//         }
//         if(mode_ == ResultMode::update) {
//           lhs_tensor.add(lblock);
//         } else if (mode_ == ResultMode::set) {
//           lhs_tensor.put(lblock);
//         } else {
//           assert(0);
//         }
//       }
//     };
//     auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
//     parallel_work(itr_first, itr_first.get_end(), lambda);
//   }

//   MapOp(const LabeledTensorType& lhs, Func func, ResultMode mode = ResultMode::set)
//       : lhs_{lhs}, func_{func}, mode_{mode} {
//     Expects(lhs_.tensor_ != nullptr);
//     Expects(lhs_.tensor_->rank() == 2);
//   }

//   LabeledTensorType lhs_;
//   Func func_;
//   ResultMode mode_;
// };

// template<typename Func, typename LabeledTensorType>
// struct MapOp<Func, LabeledTensorType, 2, 1> : public Op {
//   void execute() {
//     // std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
//     auto& lhs_tensor = *lhs_.tensor_;
//     auto& rhs1_tensor = *rhs1_.tensor_;

//     auto lambda = [&] (const TensorIndex& blockid) {
//       auto size = lhs_tensor.block_size(blockid);
//       if(!(lhs_tensor.nonzero(blockid) && lhs_tensor.spin_unique(blockid) && size > 0)) {
//         return;
//       }
//       auto bdims = lhs_tensor.block_dims(blockid);
//       auto isize = bdims[0].value();
//       auto jsize = bdims[1].value();
//       auto ioffset = TCE::offset(blockid[0]);
//       auto joffset = TCE::offset(blockid[1]);
//       if(lhs_tensor.nonzero(blockid) && isize*jsize > 0) {
//         auto lblock = lhs_tensor.alloc(blockid);
//         auto r1block = rhs1_tensor.get(blockid);
//         for(int i=0, c=0; i<isize; i++) {
//           for(int j=0; j<jsize; j++, c++) {
//             func_(ioffset+i, joffset+j, lblock.buf()[c], r1block.buf()[c]);
//           }
//         }
//         if(mode_ == ResultMode::update) {
//           lhs_tensor.add(lblock);
//         } else if (mode_ == ResultMode::set) {
//           lhs_tensor.put(lblock);
//         } else {
//           assert(0);
//         }
//       }
//     };
//     auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
//     parallel_work(itr_first, itr_first.get_end(), lambda);
//   }

//   MapOp(const LabeledTensorType& lhs, LabeledTensorType& rhs1, Func func, ResultMode mode = ResultMode::set)
//       : lhs_{lhs},
//         rhs1_{rhs1},
//         func_{func},
//         mode_{mode} {
//     Expects(lhs_.tensor_ != nullptr);
//     Expects(lhs_.tensor_->rank() == 2);
//   }

//   LabeledTensorType lhs_;
//   LabeledTensorType rhs1_;
//   Func func_;
//   ResultMode mode_;
// };

// template<typename Func, typename LabeledTensorType>
// struct MapOp<Func, LabeledTensorType, 2, 2> : public Op {
//   void execute() {
//     // std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
//     auto& lhs_tensor = *lhs_.tensor_;
//     auto& rhs1_tensor = *rhs1_.tensor_;
//     auto& rhs2_tensor = *rhs2_.tensor_;

//     auto lambda = [&] (const TensorIndex& blockid) {
//       auto size = lhs_tensor.block_size(blockid);
//       if(!(lhs_tensor.nonzero(blockid) && lhs_tensor.spin_unique(blockid) && size > 0)) {
//         return;
//       }
//       auto bdims = lhs_tensor.block_dims(blockid);
//       auto isize = bdims[0].value();
//       auto jsize = bdims[1].value();
//       auto ioffset = TCE::offset(blockid[0]);
//       auto joffset = TCE::offset(blockid[1]);
//       if(lhs_tensor.nonzero(blockid) && isize*jsize > 0) {
//         auto lblock = lhs_tensor.alloc(blockid);
//         auto r1block = rhs1_tensor.get(blockid);
//         auto r2block = rhs2_tensor.get(blockid);
//         auto ltbuf  = lblock.buf();
//         auto r1tbuf = r1block.buf();
//         auto r2tbuf = r2block.buf();
//         for(int i=0, c=0; i<isize; i++) {
//           for(int j=0; j<jsize; j++, c++) {
//             func_(ioffset+i, joffset+j, ltbuf[c], r1tbuf[c], r2tbuf[c]);
//           }
//         }
//         if(mode_ == ResultMode::update) {
//           lhs_tensor.add(lblock);
//         } else if (mode_ == ResultMode::set) {
//           lhs_tensor.put(lblock);
//         } else {
//           assert(0);
//         }
//       }
//     };
//     auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
//     parallel_work(itr_first, itr_first.get_end(), lambda);
//   }

//   MapOp(const LabeledTensorType& lhs, LabeledTensorType& rhs1, LabeledTensorType& rhs2, Func func,
//         ResultMode mode = ResultMode::set)
//       : lhs_{lhs},
//         rhs1_{rhs1},
//         rhs2_{rhs2},
//         func_{func},
//         mode_{mode}{
//     Expects(lhs_.tensor_ != nullptr);
//     Expects(lhs_.tensor_->rank() == 2);
//   }

//   LabeledTensorType lhs_;
//   LabeledTensorType rhs1_, rhs2_;
//   Func func_;
//   ResultMode mode_;
// };


// template<typename Func, typename LabeledTensorType>
// struct MapOp<Func, LabeledTensorType, 1, 1> : public Op {
//   void execute() {
//     // std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
//     auto& lhs_tensor = *lhs_.tensor_;
//     auto& rhs_tensor = *rhs_.tensor_;

//     auto lambda = [&] (const TensorIndex& blockid) {
//       auto size = lhs_tensor.block_size(blockid);
//       if(!(lhs_tensor.nonzero(blockid) && lhs_tensor.spin_unique(blockid) && size > 0)) {
//         return;
//       }
//       auto offset = TCE::offset(blockid[0]);
//       auto lblock = lhs_tensor.alloc(blockid);
//       auto rblock = rhs_tensor.get(blockid);
//       auto ltbuf = lblock.buf();
//       auto rtbuf = rblock.buf();
//       for(int i=0; i<size; i++) {
//         func_(i+offset, ltbuf[i], rtbuf[i]);
//       }
//       if(mode_ == ResultMode::update) {
//         lhs_tensor.add(lblock);
//       } else if (mode_ == ResultMode::set) {
//         lhs_tensor.put(lblock);
//       } else {
//         assert(0);
//       }
//     };
//     auto itr_first = loop_iterator(slice_indices(lhs_tensor.indices(), lhs_.label_));
//     parallel_work(itr_first, itr_first.get_end(), lambda);
//   }

//   MapOp(const LabeledTensorType& lhs, const LabeledTensorType& rhs, Func func, ResultMode mode = ResultMode::set)
//       : lhs_{lhs},
//         rhs_{rhs},
//         func_{func},
//         mode_{mode} {
//     Expects(lhs_.tensor_ != nullptr);
//     Expects(lhs_.tensor_->rank() == 1);
//     Expects(rhs_.tensor_ != nullptr);
//     Expects(rhs_.tensor_->rank() == 1);
//   }

//   LabeledTensorType lhs_, rhs_;
//   Func func_;
//   ResultMode mode_;
// };
// #endif

/////////////////////////////////////////////////////////////////////
//         scan operator
/////////////////////////////////////////////////////////////////////

/**
 * @todo Could be more general, similar to MapOp
 */
template<typename Func, typename LabeledTensorType>
struct ScanOp : public Op {
  void execute() {
    // std::cerr<<__FUNCTION__<<":"<<__LINE__<<": ScanOp\n";
    auto& tensor = *ltensor_.tensor_;
    auto lambda = [&] (const TensorIndex& blockid) {
      auto size = tensor.block_size(blockid);      
      if(!(tensor.nonzero(blockid) && tensor.spin_unique(blockid) && size > 0)) {
        return;
      }
      auto block = tensor.get(blockid);
      auto tbuf = block.buf();
      for(int i=0; i<size; i++) {
        func_(tbuf[i]);
      }
    };
    auto itr_first = loop_iterator(slice_indices(tensor.indices(), ltensor_.label_));
    parallel_work(itr_first, itr_first.get_end(), lambda);
  }

  ScanOp(const LabeledTensorType& ltensor, Func func)
      : ltensor_{ltensor},
        func_{func} {
    Expects(ltensor.tensor_ != nullptr);
  }

  LabeledTensorType ltensor_;
  Func func_;
};

// template<typename Func, typename LabeledTensorType>
// struct ScanOp<Func,LabeledTensorType,1> : public Op {
//   void execute() {
//     // std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MapOp\n";
//     auto& tensor = *ltensor_.tensor_;
//     auto lambda = [&] (const TensorIndex& blockid) {
//       auto size = tensor.block_size(blockid);
//       if(!(tensor.nonzero(blockid) && tensor.spin_unique(blockid) && size > 0)) {
//         return;
//       }
//       auto bdims = tensor.block_dims(blockid);
//       auto isize = bdims[0].value();
//       auto ioffset = TCE::offset(blockid[0]);
//       auto block = tensor.get(blockid);
//       auto tbuf = block.buf();
//       for(int i=0; i<isize; i++) {
//         func_(i+ioffset, tbuf[i]);
//       }
//     };
//     auto itr_first = loop_iterator(slice_indices(tensor.indices(), ltensor_.label_));
//     parallel_work(itr_first, itr_first.get_end(), lambda);
//   }

//   ScanOp(const LabeledTensorType& ltensor, Func func)
//       : ltensor_{ltensor},
//         func_{func} {
//     Expects(ltensor.tensor_ != nullptr);
//     Expects(ltensor.tensor_->rank() == 1);
//   }

//   LabeledTensorType ltensor_;
//   Func func_;
// };

// template<typename Func, typename LabeledTensorType>
// struct ScanOp<Func,LabeledTensorType,2> : public Op {
//   void execute() {
//     auto& tensor = *ltensor_.tensor_;
//     Expects(tensor.rank()==2);
//     auto lambda = [&] (const TensorIndex& blockid) {
//       auto size = tensor.block_size(blockid);
//       if(!(tensor.nonzero(blockid) && tensor.spin_unique(blockid) && size > 0)) {
//         return;
//       }
//       auto bdims = tensor.block_dims(blockid);
//       auto isize = bdims[0].value();
//       auto jsize = bdims[1].value();
//       auto ioffset = TCE::offset(blockid[0]);
//       auto joffset = TCE::offset(blockid[1]);
//       auto block = tensor.get(blockid);
//       auto tbuf = block.buf();
//       Expects(isize*jsize == block.size());
//       for(unsigned i=0, c=0; i<isize; i++) {
//         for(unsigned j=0; j<jsize; j++, c++) {
//           func_(i+ioffset, j+joffset, tbuf[c]);
//         }
//       }
//     };
//     auto itr_first = loop_iterator(slice_indices(tensor.indices(), ltensor_.label_));
//     parallel_work(itr_first, itr_first.get_end(), lambda);
//   }

//   ScanOp(const LabeledTensorType& ltensor, Func func)
//       : ltensor_{ltensor},
//         func_{func} {
//     Expects(ltensor.tensor_ != nullptr);
//     Expects(ltensor.tensor_->rank() == 2);
//   }

//   LabeledTensorType ltensor_;
//   Func func_;
// };

class Scheduler {
 public:
  enum class TensorStatus { invalid, allocated, deallocated, initialized };

  Scheduler(ProcGroup pg, Distribution* default_distribution,
            MemoryManager* default_memory_manager,
            Irrep default_irrep, bool default_spin_restricted)
      : default_distribution_{default_distribution},
        default_memory_manager_{default_memory_manager},
        default_irrep_{default_irrep},
        default_spin_restricted_{default_spin_restricted},
        pg_{pg} {}

  ~Scheduler() {
    for(auto &ptr_op : ops_) {
      delete ptr_op;
    }
    for(auto itr = tensors_.begin(); itr!= tensors_.end(); ++itr) {
      if(!itr->second.is_io) {
        delete itr->first;
      }
    }
  }

  template<typename T>
  Tensor<T>& tensor(const IndexInfo& iinfo, Irrep irrep, bool spin_restricted) {
    auto indices = std::get<0>(iinfo);
    auto nupper_indices = std::get<1>(iinfo);
    auto *ptensor = new Tensor<T>{indices, nupper_indices, irrep, spin_restricted};
    tensors_[ptensor] = TensorInfo{TensorStatus::invalid, false};
    return *ptensor;
  }

  template<typename T>
  Tensor<T>& tensor(const IndexInfo& iinfo) {
    return tensor<T>(iinfo, default_irrep_, default_spin_restricted_);
  }

  template<typename T, typename LabeledTensorType>
  Scheduler& operator()(SetOpEntry<T, LabeledTensorType> sop) {
    ops_.push_back(new SetOp<LabeledTensorType, T>(sop.value, sop.lhs, sop.mode));
    Expects(tensors_.find(&sop.lhs.tensor()) != tensors_.end());
    Expects(tensors_[&sop.lhs.tensor()].status == TensorStatus::allocated
            || tensors_[&sop.lhs.tensor()].status == TensorStatus::initialized);
    tensors_[&sop.lhs.tensor()].status = TensorStatus::initialized;
    return *this;
  }

  Scheduler& io() {
    return *this;
  }

  template<typename ...Args>
  Scheduler& io(TensorBase &tensor, Args& ... args) {
    Expects(tensors_.find(&tensor) == tensors_.end());
    tensors_[&tensor] = TensorInfo{TensorStatus::initialized, true};
    return io(args...);
  }

  Scheduler& output() {
    return *this;
  }

  template<typename ...Args>
  Scheduler& output(TensorBase& tensor, Args& ... args) {
    Expects(tensors_.find(&tensor) == tensors_.end());
    tensors_[&tensor] = TensorInfo{TensorStatus::allocated, true};
    return output(args...);
  }

  Scheduler& alloc() {
    return *this;
  }

  template<typename TensorType, typename ...Args>
  Scheduler& alloc(TensorType& tensor, Args& ... args) {
    Expects(tensors_.find(&tensor) != tensors_.end());
    Expects(tensors_[&tensor].status == TensorStatus::invalid ||
            tensors_[&tensor].status == TensorStatus::deallocated);
    tensors_[&tensor].status = TensorStatus::allocated;
    ops_.push_back(new AllocOp<TensorType>(tensor, pg_, default_distribution_, default_memory_manager_));
    return alloc(args...);
  }

  Scheduler& dealloc() {
    return *this;
  }

  template<typename TensorType, typename ...Args>
  Scheduler& dealloc(TensorType& tensor, Args& ... args) {
    Expects(tensors_.find(&tensor) != tensors_.end());
    Expects(tensors_[&tensor].status == TensorStatus::allocated ||
            tensors_[&tensor].status == TensorStatus::initialized);
    tensors_[&tensor].status = TensorStatus::deallocated;
    ops_.push_back(new DeallocOp<TensorType>(&tensor));
    return dealloc(args...);
  }


  template<typename T, typename LabeledTensorType>
  Scheduler& operator()(AddOpEntry<T, LabeledTensorType> aop) {
    Expects(tensors_.find(&aop.lhs.tensor()) != tensors_.end());
    Expects(tensors_.find(&aop.rhs.tensor()) != tensors_.end());
    Expects(tensors_[&aop.rhs.tensor()].status == TensorStatus::initialized);
    Expects(tensors_[&aop.lhs.tensor()].status == TensorStatus::initialized
            || (aop.mode==ResultMode::set
                && tensors_[&aop.lhs.tensor()].status==TensorStatus::allocated));
    tensors_[&aop.lhs.tensor()].status = TensorStatus::initialized;
    ops_.push_back(new AddOp<LabeledTensorType, T>(aop.alpha, aop.lhs, aop.rhs, aop.mode));    
    return *this;
  }

  template<typename T, typename LabeledTensorType>
  Scheduler& operator()(MultOpEntry<T, LabeledTensorType> aop) {
    Expects(tensors_.find(&aop.lhs.tensor()) != tensors_.end());
    Expects(tensors_.find(&aop.rhs1.tensor()) != tensors_.end());
    Expects(tensors_.find(&aop.rhs2.tensor()) != tensors_.end());
    Expects(tensors_[&aop.rhs1.tensor()].status == TensorStatus::initialized);
    Expects(tensors_[&aop.rhs2.tensor()].status == TensorStatus::initialized);
    Expects(tensors_[&aop.lhs.tensor()].status == TensorStatus::initialized
            || (aop.mode==ResultMode::set
                && tensors_[&aop.lhs.tensor()].status==TensorStatus::allocated));
    tensors_[&aop.lhs.tensor()].status = TensorStatus::initialized;
    ops_.push_back(new MultOp<LabeledTensorType, T>(aop.alpha, aop.lhs, aop.rhs1, aop.rhs2, aop.mode));
    return *this;
  }

  template<typename Func, typename LabeledTensorType>
  Scheduler& operator()(LabeledTensorType lhs, Func func, ResultMode mode = ResultMode::set) {
    Expects(tensors_.find(&lhs.tensor()) != tensors_.end());
    Expects(tensors_[&lhs.tensor()].status == TensorStatus::initialized
            || (mode==ResultMode::set
                && tensors_[&lhs.tensor()].status==TensorStatus::allocated));
    tensors_[&lhs.tensor()].status = TensorStatus::initialized;
    // ops_.push_back(new MapOp<Func,LabeledTensorType,0,0>(lhs, func, mode));
    ops_.push_back(new MapOp<LabeledTensorType, Func, 0>(lhs, func, mode));
    return *this;
  }

  template<typename Func, typename LabeledTensorType>
  Scheduler& sop(LabeledTensorType lhs, Func func) {
    Expects(tensors_.find(&lhs.tensor()) != tensors_.end());
    Expects(tensors_[&lhs.tensor()].status == TensorStatus::initialized);
    tensors_[&lhs.tensor()].status = TensorStatus::initialized;
    ops_.push_back(new ScanOp<Func,LabeledTensorType>(lhs, func));
    return *this;
  }

  void execute() {
    for(auto &op_ptr: ops_) {
      op_ptr->execute();
    }
  }

  void clear() {
    ops_.clear();
    tensors_.clear();
  }

  Distribution* default_distribution() {
    return default_distribution_;    
  }

  MemoryManager* default_memory_manager() {
    return default_memory_manager_;
  }

  ProcGroup pg() const {
    return pg_;
  }
  
 private:
  struct TensorInfo {
    TensorStatus status;
    bool is_io;
  };
  Distribution* default_distribution_;
  MemoryManager* default_memory_manager_;
  Irrep default_irrep_;
  bool default_spin_restricted_;
  ProcGroup pg_;
  std::map<TensorBase*,TensorInfo> tensors_;
  std::vector<Op*> ops_;
};


class ExecutionContext {
 public:
  ExecutionContext(ProcGroup pg, Distribution* default_distribution,
                   MemoryManager* default_memory_manager,
                   Irrep default_irrep, bool default_spin_restricted)
      : pg_{pg},
        default_distribution_{default_distribution},
        default_memory_manager_{default_memory_manager},
        default_irrep_{default_irrep},
        default_spin_restricted_{default_spin_restricted} {}

  Scheduler scheduler() {
    return Scheduler{pg_,
          default_distribution_,
          default_memory_manager_,
          default_irrep_,
          default_spin_restricted_};
  }

  void allocate() {
    //no-op
  }

  template<typename T, typename ...Args>
  void allocate(Tensor<T>& tensor, Args& ... tensor_list) {
    tensor.alloc(pg_, default_distribution_, default_memory_manager_);
    allocate(tensor_list...);
  }


  static void deallocate() {
    //no-op
  }

  template<typename T, typename ...Args>
  static void deallocate(Tensor<T>& tensor, Args& ... tensor_list) {
    tensor.dealloc();
    deallocate(tensor_list...);
  }

  ProcGroup pg() const {
    return pg_;
  }

  Distribution* distribution() const {
    return default_distribution_;
  }

  MemoryManager* memory_manager() const {
    return default_memory_manager_;
  }

  Irrep irrep() const {
    return  default_irrep_;
  }

  bool is_spin_restricted() const {
    return default_spin_restricted_;
  }

 private:
  ProcGroup pg_;
  Distribution* default_distribution_;
  MemoryManager* default_memory_manager_;
  Irrep default_irrep_;
  bool default_spin_restricted_;
}; // class ExecutionContext

///////////////////////////////////////////////////////////////////////
// other operations
//////////////////////////////////////////////////////////////////////

template<typename T>
inline void
assert_zero(Scheduler& sch, Tensor<T>& tc, double threshold = 1.0e-12) {
  auto lambda = [&] (auto &val) {
    //    std::cout<<"assert_zero. val="<<val<<std::endl;
    Expects(std::abs(val) < threshold);
  };
  sch.io(tc)
      .sop(tc(), lambda)
      .execute();
}

template<typename LabeledTensorType, typename T>
inline void
assert_equal(Scheduler& sch, LabeledTensorType tc, T value, double threshold = 1.0e-12) {
  auto lambda = [&] (auto &val) {
    Expects(std::abs(val - value) < threshold);
  };
  sch.io(tc.tensor())
      .sop(tc, lambda)
      .execute();
}

template<typename LabeledTensorType>
inline void
tensor_print(Scheduler& sch, LabeledTensorType ltensor) {
  sch.io(ltensor.tensor())
      .sop(ltensor, [] (auto &ival) {
          std::cout<<ival<<" ";
        })
      .execute();
}

///////////////////////////////////////////////////////////////////////
// implementation of operators
//////////////////////////////////////////////////////////////////////

//-----------------------support routines


template<typename T>
inline TensorVec<TensorVec<TensorLabel>>
summation_labels(const LabeledTensor<T>& /*ltc*/,
                 const LabeledTensor<T>& lta,
                 const LabeledTensor<T>& ltb) {
  return group_partition(lta.tensor_->indices(), lta.label_,
                         ltb.tensor_->indices(), ltb.label_);

}

template<typename T>
inline std::pair<TensorVec<SymmGroup>,TensorLabel>
summation_indices(const LabeledTensor<T>& /*ltc*/,
                  const LabeledTensor<T>& lta,
                  const LabeledTensor<T>& ltb) {
  auto aindices = flatten(lta.tensor_->indices());
  //auto bindices = flatten(ltb.tensor_.indices());
  auto alabels = group_labels(lta.tensor_->indices(), lta.label_);
  auto blabels = group_labels(ltb.tensor_->indices(), ltb.label_);
  TensorVec<SymmGroup> ret_indices;
  TensorLabel sum_labels;
  int apos = 0;
  for (auto &alg : alabels) {
    for (auto &blg : blabels) {
      SymmGroup sg;
      for (auto &a : alg) {
        int apos1 = 0;
        for (auto &b : blg) {
          if (a == b) {
            sg.push_back(aindices[apos + apos1]);
            sum_labels.push_back(a);
          }
        }
        apos1++;
      }
      if (sg.size() > 0) {
        ret_indices.push_back(sg);
      }
    }
    apos += alg.size();
  }
  return {ret_indices, sum_labels};
}

template<typename T>
inline TensorVec<TensorVec<TensorLabel>>
nonsymmetrized_external_labels(const LabeledTensor<T>& ltc,
                               const LabeledTensor<T>& lta) {
  auto ca_labels = group_partition(ltc.tensor_->indices(), ltc.label_,
                                   lta.tensor_->indices(), lta.label_);

  TensorVec<TensorVec<TensorLabel>> ret_labels;
  for(unsigned i=0; i<ca_labels.size(); i++)  {
    Expects(ca_labels[i].size() > 0);
    ret_labels.push_back(TensorVec<TensorLabel>());
    ret_labels.back().insert_back(ca_labels[i].begin(), ca_labels[i].end());
  }
  return ret_labels;
}

/**
 * @todo Specify where symmetrization is allowed and what indices in
 * the input tensors can form a symmetry group (or go to distinct
 * groups) in the output tensor.
 */
template<typename T>
inline TensorVec<TensorVec<TensorLabel>>
nonsymmetrized_external_labels(const LabeledTensor<T>& ltc,
                               const LabeledTensor<T>& lta,
                               const LabeledTensor<T>& ltb) {
  auto ca_labels = group_partition(ltc.tensor_->indices(), ltc.label_,
                                   lta.tensor_->indices(), lta.label_);
  auto cb_labels = group_partition(ltc.tensor_->indices(), ltc.label_,
                                   ltb.tensor_->indices(), ltb.label_);
  Expects(ca_labels.size() == cb_labels.size());

  TensorVec<TensorVec<TensorLabel>> ret_labels;
  for(unsigned i=0; i<ca_labels.size(); i++)  {
    Expects(ca_labels[i].size() + cb_labels[i].size() > 0);
    ret_labels.push_back(TensorVec<TensorLabel>());
    if(ca_labels[i].size() > 0) {
      ret_labels.back().insert_back(ca_labels[i].begin(), ca_labels[i].end());
    }
    if(cb_labels[i].size() > 0) {
      ret_labels.back().insert_back(cb_labels[i].begin(), cb_labels[i].end());
    }
  }
  return ret_labels;
}

template<typename T>
inline TensorVec<TensorVec<TensorLabel>>
symmetrized_external_labels(const LabeledTensor<T>& ltc,
                            const LabeledTensor<T>&  /*lta*/,
                            const LabeledTensor<T>&  /*ltb*/) {
  TensorVec<TensorLabel> ret {ltc.label_};
  return {ret};
}


template<typename T>
inline ProductIterator<TriangleLoop>
nonsymmetrized_iterator(const LabeledTensor<T>& ltc,
                        const LabeledTensor<T>& lta,
                        const LabeledTensor<T>& ltb) {
  auto part_labels = nonsymmetrized_external_labels(ltc ,lta, ltb);
  //auto flat_labels = flatten(flatten(part_labels));
  // std::map<IndexLabel, DimType> dim_of_label;

  // auto cflindices = flatten(ltc.tensor_->indices());
  // for(unsigned i=0; i<ltc.label_.size(); i++) {
  //   dim_of_label[ltc.label_[i]] = cflindices[i];
  // }
  // auto aflindices = flatten(lta.tensor_->indices());
  // for(unsigned i=0; i<lta.label_.size(); i++) {
  //   dim_of_label[lta.label_[i]] = aflindices[i];
  // }
  TensorVec<TriangleLoop> tloops, tloops_last;
  // std::cout<<__FUNCTION__<<"num_parts="<<part_labels.size()<<std::endl;
  for(auto dim_grps: part_labels) {
    for(auto lbl: dim_grps) {
    // std::cout<<__FUNCTION__<<"part="<< lbl <<std::endl;
      if(lbl.size() > 0) {
        BlockDim lo, hi;
        std::tie(lo, hi) = tensor_index_range(lbl[0].dt);
        tloops.push_back(TriangleLoop(lbl.size(), lo, hi));
        tloops_last.push_back(tloops.back().get_end());
      }
    }
  }
  if(tloops.size()==0){
    tloops.push_back(TriangleLoop{});
    tloops_last.push_back(tloops.back().get_end());
  }
  return ProductIterator<TriangleLoop>(tloops, tloops_last);
}


/**
 * @todo abstract the two copy_symmetrizer versions into one function
 * with the logic and two interfaces
 */
template<typename T>
inline TensorVec<CopySymmetrizer>
copy_symmetrizer(const LabeledTensor<T>& ltc,
                 const LabeledTensor<T>& lta,
                 const LabeledTensor<T>& ltb,
                 const LabelMap<BlockDim>& lmap) {
  auto part_labels = nonsymmetrized_external_labels(ltc ,lta, ltb);
  TensorVec<CopySymmetrizer> csv;
  for(auto lbls: part_labels) {
    Expects(lbls.size()>0 && lbls.size() <= 2);

    TensorLabel lbl(lbls[0].begin(), lbls[0].end());
    if(lbls.size() == 2 ) {
      lbl.insert_back(lbls[1].begin(), lbls[1].end());
    }

    auto size = lbl.size();
    Expects(size > 0);
    Expects(size <=2); // @todo implement other cases

    auto blockid = lmap.get_blockid(lbl);
    auto uniq_blockid{blockid};
    //find unique block
    std::sort(uniq_blockid.begin(), uniq_blockid.end());
    Expects(size > 0);
    //std::cout<<"CONSTRUCTING COPY SYMMETRIZER FOR LABELS="<<lbl<<std::endl;
    csv.push_back(CopySymmetrizer{size, lbls[0].size(), lbl, blockid, uniq_blockid});
  }
  if (csv.empty()) {
    csv.push_back(CopySymmetrizer{});
  }
  return csv;
}

inline TensorVec<TensorLabel>
group_labels(const TensorLabel& label,
             const TensorIndex& group_sizes) {
  TensorVec<TensorLabel> ret;
  int pos = 0;
  for(auto grp: group_sizes) {
    ret.push_back(TensorLabel(grp.value()));
    std::copy_n(label.begin() + pos, grp.value(), ret.back().begin());
    pos += grp.value();
  }
  return ret;
}

inline TensorVec<TensorVec<TensorLabel>>
compute_extra_symmetries(const TensorLabel& lhs_label,
                         const TensorIndex& lhs_group_sizes,
                         const TensorLabel& rhs_label,
                         const TensorIndex& rhs_group_sizes) {
  Expects(lhs_label.size() == lhs_group_sizes.size());
  Expects(rhs_label.size() == rhs_group_sizes.size());
  Expects(lhs_label.size() == rhs_label.size());

  auto lhs_label_groups = group_labels(lhs_label, lhs_group_sizes);
  auto rhs_label_groups = group_labels(rhs_label, rhs_group_sizes);

  TensorVec<TensorVec<TensorLabel>> ret_labels;
  for (auto &glhs : lhs_label_groups) {
    TensorVec<TensorLabel> ret_group;
    for (auto &grhs : rhs_label_groups) {
      auto lbls = intersect(glhs, grhs);
      if (lbls.size() > 0) {
        ret_group.push_back(lbls);
      }
    }
    ret_labels.push_back(ret_group);
  }
  return ret_labels;
}


template<typename T>
inline TensorVec<CopySymmetrizer>
copy_symmetrizer(const LabeledTensor<T>& ltc,
                 const LabeledTensor<T>& lta,
                 const LabelMap<BlockDim>& lmap) {
  auto part_labels = nonsymmetrized_external_labels(ltc ,lta);
  TensorVec<CopySymmetrizer> csv;
  for(auto lbls: part_labels) {
    Expects(lbls.size()>0 && lbls.size() <= 2);

    TensorLabel lbl(lbls[0].begin(), lbls[0].end());
    if(lbls.size() == 2 ) {
      lbl.insert_back(lbls[1].begin(), lbls[1].end());
    }

    auto size = lbl.size();
    Expects(size > 0);
    Expects(size <=2); // @todo implement other cases

    auto blockid = lmap.get_blockid(lbl);
    auto uniq_blockid{blockid};
    //find unique block
    std::sort(uniq_blockid.begin(), uniq_blockid.end());
    Expects(size > 0);
    std::cout<<"CONSTRUCTING COPY SYMMETRIZER FOR LABELS="<<lbl
             <<" size="<<size
             <<" lbl0="<<lbls[0]
             <<" blockid="<<blockid
             <<" uniq_blockid="<<uniq_blockid
             <<std::endl;
    csv.push_back(CopySymmetrizer{size, lbls[0].size(), lbl, uniq_blockid, uniq_blockid});
    Expects(csv.back().begin() != csv.back().end());
  }
  if (csv.empty()) {
    csv.push_back(CopySymmetrizer{});
    Expects(csv.back().begin() != csv.back().end());
  }
  return csv;
}



inline ProductIterator<CopySymmetrizer::Iterator>
copy_iterator(const TensorVec<CopySymmetrizer>& sitv) {
  TensorVec<CopySymmetrizer::Iterator> itrs_first, itrs_last;
  for(auto &sit: sitv) {
    Expects(sit.begin() != sit.end());
    itrs_first.push_back(sit.begin());
    itrs_last.push_back(sit.end());
    Expects(itrs_first.back().itr_size() == itrs_last.back().itr_size());
    Expects(itrs_first.back().itr_size() == sit.group_size_);
  }
  return {itrs_first, itrs_last};
}


//-----------------------op execute routines

template<typename T, typename LabeledTensorType>
inline void
SetOp<T,LabeledTensorType>::execute() {
  using T1 = typename LabeledTensorType::element_type;
  // std::cerr<<"Calling setop :: execute"<<std::endl;
  auto& tensor = *lhs_.tensor_;
  auto lambda = [&] (const TensorIndex& blockid) {
    auto size = tensor.block_size(blockid);
    if(!(tensor.nonzero(blockid) && tensor.spin_unique(blockid) && size > 0)) {
      return;
    }
    auto block = tensor.alloc(blockid);
    auto tbuf = reinterpret_cast<T1*>(block.buf());
    auto value = static_cast<T1>(value_);
    for(int i=0; i<size; i++) {
      tbuf[i] = value;
    }
    if(mode_ == ResultMode::update) {
      tensor.add(block.blockid(), block);
    } else if (mode_ == ResultMode::set) {
      tensor.put(block.blockid(), block);
    } else {
      assert(0);
    }
  };
  auto itr_first = loop_iterator(slice_indices(tensor.indices(), lhs_.label_));
  parallel_work(itr_first, itr_first.get_end(), lambda);
}

//--------------- new symmetrization routines

inline TensorLabel
comb_bv_to_label(const TensorVec<int>& comb_itr, const TensorLabel& tlabel) {
  assert(comb_itr.size() == tlabel.size());
  auto n = comb_itr.size();
  TensorLabel left, right;
  for(size_t i=0; i<n; i++) {
    if(comb_itr[i] == 0) {
      left.push_back(tlabel[i]);
    }
    else {
      right.push_back(tlabel[i]);
    }
  }
  TensorLabel ret{left};
  ret.insert_back(right.begin(), right.end());
  return ret;
}

/**
   @note Find first index of value in lst[lo,hi). Index hi is exclusive.
 */
template<typename T>
size_t find_first_index(const TensorVec<T> lst, size_t lo, size_t hi, T value) {
  auto i= lo;
  for(; i<hi; i++) {
    if (lst[i] == value) {
      return i;
    }
  }
  return i;
}

/**
   @note Find last index of value in lst[lo,hi). Index hi is exclusive.
 */
template<typename T>
size_t find_last_index(const TensorVec<T> lst, size_t lo, size_t hi, T value) {
  auto i = hi - 1;
  for( ;i >= lo; i--) {
    if (lst[i] == value) {
      return i;
    }
  }
  return i;
}

/**
   @note comb_itr is assumed to be a vector of 0s and 1s
 */
inline bool
is_unique_combination(const TensorVec<int>& comb_itr, const TensorIndex& lval) {
  Expects(std::is_sorted(lval.begin(), lval.end()));
  Expects(comb_itr.size() == lval.size());
  for(auto ci: comb_itr) {
    Expects(ci ==0 || ci == 1);
  }
  auto n = lval.size();
  size_t i = 0;
  while(i < n) {
    auto lo = i;
    auto hi = i+1;
    while(hi < n && lval[hi] == lval[lo]) {
      hi += 1;
    }
    auto first_one = find_first_index(comb_itr, lo, hi, 1);
    auto last_zero = find_last_index(comb_itr, lo, hi, 0);
    if(first_one < last_zero) {
      return false;
    }
    i = hi;
  }
  return true;
}

class SymmetrizerNew {
 public:
  using element_type = IndexLabel;
  SymmetrizerNew(const LabelMap<BlockDim>& lmap,
                 const TensorLabel& olabels,
                 size_t nsymm_indices)
      : lmap_{lmap},
        olabels_{olabels},
        nsymm_indices_{nsymm_indices},
        done_{false} {
          reset();
        }

  bool has_more() const {
    return !done_;
  }

  void reset() {
    done_ = false;
    auto n = olabels_.size();
    auto k = nsymm_indices_;
    Expects(k>=0 && k<=n);
    comb_itr_.resize(n);
    std::fill_n(comb_itr_.begin(), k, 0);
    std::fill_n(comb_itr_.begin()+k, n-k, 1);
    Expects(comb_itr_.size() == olabels_.size());
    olval_ = lmap_.get_blockid(olabels_);
  }
  
  TensorLabel get() const {
    Expects(comb_itr_.size() == olabels_.size());
    return comb_bv_to_label(comb_itr_, olabels_);
  }

  size_t itr_size() const {
    return comb_itr_.size();
  }

  void next() {
    //done_ = true; return; //@BUG
    //std::cout<<"Entered. SymmetrizerNew, next"<<std::endl;
    do {
      // std::cout<<"Comb itr="<<comb_itr_<<std::endl;
      // std::cout<<"olval="<<olval_<<std::endl;
      // std::cout<<"done="<<done_<<std::endl;
      done_ = !std::next_permutation(comb_itr_.begin(), comb_itr_.end());
      // std::cout<<"Comb itr="<<comb_itr_<<std::endl;
      // std::cout<<"olval="<<olval_<<std::endl;
      // std::cout<<"done="<<done_<<std::endl;
    } while (!done_ && !is_unique_combination(comb_itr_, olval_));
    //std::cout<<"Exited. SymmetrizerNew, next"<<std::endl;
    Expects(comb_itr_.size() == olabels_.size());
  }
  
 private:
  const LabelMap<BlockDim> lmap_;
  TensorLabel olabels_;
  TensorIndex olval_;
  TensorVec<int> comb_itr_;
  size_t nsymm_indices_;
  bool done_;
};


class CopySymmetrizerNew {
 public:
  using element_type = IndexLabel;
  CopySymmetrizerNew(const LabelMap<BlockDim>& lmap,
                     const TensorLabel& olabels,
                     const TensorIndex& cur_olval,
                     size_t nsymm_indices)
      : lmap_{lmap},
        olabels_{olabels},
        cur_olval_{cur_olval},
        nsymm_indices_{nsymm_indices},
        done_{false} {
          Expects(nsymm_indices>=0 && nsymm_indices<=olabels.size());
          reset();
        }
  
  bool has_more() const {
    return !done_;
  }

  void reset() {
    done_ = false;
    auto n = olabels_.size();
    auto k = nsymm_indices_;
    comb_itr_.resize(n);
    std::fill_n(comb_itr_.begin(), k, 0);
    std::fill_n(comb_itr_.begin()+k, n-k, 1);
    progress();    
  }

  TensorLabel get() const {
    return cur_label_;
  }

  size_t itr_size() const {
    return comb_itr_.size();
  }

  void progress() {
    while(!done_) {
      cur_label_ = comb_bv_to_label(comb_itr_, olabels_);
      auto lval = lmap_.get_blockid(cur_label_);
      if(std::equal(cur_olval_.begin(), cur_olval_.end(),
                    lval.begin(), lval.end())) {
        break;
      }
      done_ = !std::next_permutation(comb_itr_.begin(), comb_itr_.end());
    }
  }

  void next() {
    done_ = !std::next_permutation(comb_itr_.begin(), comb_itr_.end());
    progress();
  }
  
 private:
  const LabelMap<BlockDim> lmap_;
  TensorLabel olabels_;
  TensorLabel cur_label_;
  TensorIndex cur_olval_;
  TensorVec<int> comb_itr_;
  size_t nsymm_indices_;
  bool done_;
};

template<typename Itr>
class SymmIterator {
 public:
  SymmIterator(const std::vector<Itr>& itrs)
      : itrs_{itrs},
        done_{false} {
          reset();
        }

  void reset() {
    for(auto& it: itrs_) {
      it.reset();
      Expects(it.has_more());
    }    
  }
  
  size_t itr_size() const {
    size_t ret = 0;
    for(const auto& it: itrs_) {
      ret += it.size();
    }
    return ret;
  }

  bool has_more() {
    return !done_;
  }

  TensorLabel get() const {
    TensorLabel ret;
    for(const auto& it: itrs_) {
      auto vtmp = it.get();
      ret.insert_back(vtmp.begin(), vtmp.end());
    }
    return ret;
  }

  void next() {
    int i = itrs_.size()-1;
    for(; i>=0; i--) {
      itrs_[i].next();
      if (itrs_[i].has_more()) {
        //std::cout<<"ACTION NEXT HAS MORE"<<std::endl;
        break;
      }
      itrs_[i].reset();
    }
    if (i<0) {
      done_ = true;
    }
  }
  
 private:
  std::vector<Itr> itrs_;
  bool done_;
};

inline SymmIterator<SymmetrizerNew>
symmetrization_iterator(LabelMap<BlockDim> lmap,
                        const std::vector<TensorLabel>& grps,
                        const std::vector<size_t> nsymm_indices) {
  Expects(grps.size() == nsymm_indices.size());
  std::vector<SymmetrizerNew> symms;
  for(size_t i=0; i<grps.size(); i++) {
    symms.emplace_back(lmap, grps[i], nsymm_indices[i]);
  }
  return {symms};
}

inline SymmIterator<CopySymmetrizerNew>
copy_symmetrization_iterator(LabelMap<BlockDim> lmap,
                             const std::vector<TensorLabel>& grps,
                             const std::vector<TensorIndex>& lvals,
                             const std::vector<size_t> nsymm_indices) {
  Expects(grps.size() == nsymm_indices.size());
  std::vector<CopySymmetrizerNew> symms;
  for(size_t i=0; i<grps.size(); i++) {
    symms.emplace_back(lmap, grps[i], lvals[i], nsymm_indices[i]);
  }
  return {symms};
}

/**
   @note assume one group in output (ltc) is atmost split into two
   groups in input (lta) and that, for given group, there is either
   symmetrization or unsymmetrization, but not both.
 */
template<typename LabeledTensorType>
inline SymmIterator<SymmetrizerNew>
symmetrization_iterator(LabelMap<BlockDim>& lmap,
                        const LabeledTensorType& ltc,
                        const LabeledTensorType& lta) {
  std::vector<TensorLabel> cgrps_vec;
  auto cgrps = group_labels(ltc.tensor_->indices(),
                            ltc.label_);
  for(const auto& cgrp: cgrps) {
    cgrps_vec.push_back(cgrp);
  }

  auto cgrp_parts = group_partition(ltc.tensor_->indices(),
                                    ltc.label_,
                                    lta.tensor_->indices(),
                                    lta.label_);
  std::vector<size_t> nsymm_indices;
  for(const auto& csgp: cgrp_parts) {
    Expects(csgp.size() >=0 && csgp.size() <= 2);
    nsymm_indices.push_back(csgp[0].size());
  }
  return symmetrization_iterator(lmap, cgrps_vec, nsymm_indices);
}

/**
   @note assume one group in output (ltc) is atmost split into two
   groups in input (lta) and that, for given group, there is either
   symmetrization or unsymmetrization, but not both.
 */
template<typename LabeledTensorType>
inline SymmIterator<CopySymmetrizerNew>
copy_symmetrization_iterator(LabelMap<BlockDim>& lmap,
                             const LabeledTensorType& ltc,
                             const LabeledTensorType& lta,
                             const TensorIndex& cur_clval) {
  std::vector<TensorLabel> cgrps_vec;
  auto cgrps = group_labels(ltc.tensor_->indices(),
                            ltc.label_);
  for(const auto& cgrp: cgrps) {
    cgrps_vec.push_back(cgrp);
  }
  
  auto cgrp_parts = group_partition(ltc.tensor_->indices(),
                                    ltc.label_,
                                    lta.tensor_->indices(),
                                    lta.label_);
  std::vector<size_t> nsymm_indices;
  for(const auto& csgp: cgrp_parts) {
    Expects(csgp.size() >=0 && csgp.size() <= 2);
    nsymm_indices.push_back(csgp[0].size());
  }
  std::vector<TensorIndex> clvals;
  int i = 0;
  for(const auto& csg: ltc.tensor_->indices()) {
    clvals.push_back(TensorIndex{cur_clval.begin()+i,
            cur_clval.begin()+i+csg.size()});
    i += csg.size();
  }
  
  return copy_symmetrization_iterator(lmap, cgrps_vec, clvals, nsymm_indices);
}

template<typename LabeledTensorType>
double
compute_symmetrization_factor(const LabeledTensorType& ltc,
                              const LabeledTensorType& lta) {
  auto cgrp_parts = group_partition(ltc.tensor_->indices(),
                                    ltc.label_,
                                    lta.tensor_->indices(),
                                    lta.label_);
  double ret = 1.0;
  for(const auto& csgp: cgrp_parts) {
    Expects(csgp.size() >=0 && csgp.size() <= 2);
    int n = csgp[0].size() + csgp[1].size();
    int r = csgp[0].size();
    ret *= factorial(n) / (factorial(r) * factorial(n-r));
  }
  return ret;
}
    

#if 0
// with symmetrization, no unsymmetrization. does not work
template<typename T, typename LabeledTensorType>
inline void
AddOp<T, LabeledTensorType>::execute() {
  using T1 = typename LabeledTensorType::element_type;
  // std::cerr<<__FUNCTION__<<":"<<__LINE__<<": AddOp\n";
  const LabeledTensor<T1>& lta = rhs_;
  const LabeledTensor<T1>& ltc = lhs_;
  Tensor<T1>& ta = *lta.tensor_;
  Tensor<T1>& tc = *ltc.tensor_;
  auto aitr = loop_iterator(slice_indices(ta.indices(), lta.label_));
  auto lambda = [&] (const TensorIndex& ablockid) {
    std::cout<<"---tammx assign. ablockid"<<ablockid<<std::endl;
    size_t dima = ta.block_size(ablockid);
    if(!(ta.nonzero(ablockid) && ta.spin_unique(ablockid) && dima > 0)) {
      return;
    }
    auto label_map = LabelMap<BlockDim>().update(lta.label_, ablockid);
    auto cblockid = label_map.get_blockid(ltc.label_);
    auto abp = ta.get(ablockid);
    auto csbp = tc.alloc(tc.find_unique_block(cblockid));
    std::cout<<"---tammx assign. cblockid"<<csbp.blockid()<<std::endl;
    csbp() = T(0);
    auto copy_symm = copy_symmetrizer(ltc, lta, label_map);
    auto copy_itr = copy_iterator(copy_symm);
    auto copy_itr_last = copy_itr.get_end();
    auto copy_label = TensorLabel{ltc.label_};

    for(auto &citr = copy_itr; citr != copy_itr_last; ++citr) {
      auto cperm_label = *citr;
      auto num_inversions = perm_count_inversions(perm_compute(copy_label, cperm_label));
      Sign sign = (num_inversions%2) ? -1 : 1;
      auto perm_comp = perm_apply(cperm_label, perm_compute(ltc.label_, lta.label_));
      csbp(copy_label) += sign * alpha_ * abp(perm_comp);
      std::cout<<"----AddOp. execute. csbp["<<csbp.blockid()<<"]("<<copy_label<<") "
               <<"="
               << sign << "*" << alpha_ << "*"
               << "abp["<<abp.blockid()<<"] ("<<perm_comp<<")\n";
    }
    std::cout<<"AddOp. csbp="<<csbp.buf()<<std::endl;
    if(mode_ == ResultMode::update) {
      tc.add(csbp.blockid(), csbp);
    } else {
      tc.put(csbp.blockid(), csbp);
    }
  };
  parallel_work(aitr, aitr.get_end(), lambda);
}

#elif 0
// no symmetrization, no unsymmetrization
template<typename T, typename LabeledTensorType>
inline void
AddOp<T, LabeledTensorType>::execute() {
  using T1 = typename LabeledTensorType::element_type;
  // std::cerr<<__FUNCTION__<<":"<<__LINE__<<": AddOp\n";
  const LabeledTensor<T1>& lta = rhs_;
  const LabeledTensor<T1>& ltc = lhs_;
  Tensor<T1>& ta = *lta.tensor_;
  Tensor<T1>& tc = *ltc.tensor_;
  auto aitr = loop_iterator(slice_indices(ta.indices(), lta.label_));
  auto lambda = [&] (const TensorIndex& ablockid) {
    std::cout<<"---tammx assign. ablockid"<<ablockid<<std::endl;
    size_t dima = ta.block_size(ablockid);
    if(!(ta.nonzero(ablockid) && ta.spin_unique(ablockid) && dima > 0)) {
      return;
    }
    auto label_map = LabelMap<BlockDim>().update(lta.label_, ablockid);
    auto cblockid = label_map.get_blockid(ltc.label_);
    auto cuniq_blockid = tc.find_unique_block(cblockid);
    Sign sign;
    TensorPerm cperm;
    std::tie(cperm, sign) = tc.compute_sign_from_unique_block(cblockid);
    auto cuniq_label = perm_apply(ltc.label_, cperm);

    auto abp = ta.get(ablockid);
    auto csbp = tc.alloc(cuniq_blockid);
    csbp() = 0;
#if 0
    //@note this works. Commented to avoid performance cost. But kept
    //to illustrate the logic.
    auto cbp = tc.alloc(cblockid);
    cbp(ltc.label_) += alpha_ * abp(lta.label_);
    csbp(cuniq_label) += sign * cbp(ltc.label_);
#else
    csbp(cuniq_label) += alpha_ * sign * abp(lta.label_);
#endif
    std::cout<<"----AddOp. execute. csbp["<<csbp.blockid()<<"]("<<cuniq_label<<") "
    <<"="
    << sign << "*" << alpha_ << "*"
    << "abp["<<abp.blockid()<<"] ("<<ltc.label_<<")\n";
    if(mode_ == ResultMode::update) {
      tc.add(csbp.blockid(), csbp);
    } else {
      tc.put(csbp.blockid(), csbp);
    }
  };
  parallel_work(aitr, aitr.get_end(), lambda);
}
#else
// symmetrization or unsymmetrization, but not both in one symmetry group
template<typename T, typename LabeledTensorType>
inline void
AddOp<T, LabeledTensorType>::execute() {
  using T1 = typename LabeledTensorType::element_type;
  const LabeledTensor<T1>& lta = rhs_;
  const LabeledTensor<T1>& ltc = lhs_;
  const auto &clabel = ltc.label_;
  const auto &alabel = lta.label_;
  Tensor<T1>& ta = *lta.tensor_;
  Tensor<T1>& tc = *ltc.tensor_;
  double symm_factor = compute_symmetrization_factor(ltc, lta);
  //std::cout<<"===symm factor="<<symm_factor<<std::endl;
  auto citr = loop_iterator(slice_indices(tc.indices(), ltc.label_));
  auto lambda = [&] (const TensorIndex& cblockid) {
    //std::cout<<"---tammx assign. cblockid"<<cblockid<<std::endl;
    size_t dimc = tc.block_size(cblockid);
    if(!(tc.nonzero(cblockid) && tc.spin_unique(cblockid) && dimc > 0)) {
      return;
    }
    //std::cout<<"---tammx assign. ACTION ON cblockid"<<cblockid<<std::endl;
    auto label_map = LabelMap<BlockDim>().update(ltc.label_, cblockid);
    auto sit = symmetrization_iterator(label_map,ltc, lta);
    for(; sit.has_more(); sit.next()) {
      TensorLabel cur_clbl = sit.get();
      //std::cout<<"ACTION cur_clbl="<<cur_clbl<<std::endl;
      auto cur_cblockid = label_map.get_blockid(cur_clbl);      
      auto ablockid = LabelMap<BlockDim>().update(ltc.label_, cur_cblockid).get_blockid(alabel);
      auto abp = ta.get(ablockid);
      //std::cout<<"ACTION ablockid="<<ablockid<<std::endl;

      auto csbp = tc.alloc(cur_cblockid);
      csbp() = 0;
      csbp(ltc.label_) += alpha_ * symm_factor * abp(lta.label_);

      auto cbp = tc.alloc(cblockid);
      cbp() = 0;
      auto csit = copy_symmetrization_iterator(label_map, ltc, lta, cur_cblockid);
      for(TensorLabel csym_clbl = csit.get(); csit.has_more(); csit.next(), csym_clbl = csit.get()) {
        int csym_sign = (perm_count_inversions(perm_compute(cur_clbl, csym_clbl)) % 2) ? -1 : 1;
        //std::cout<<"===csym sign="<<csym_sign<<std::endl;      
        cbp(ltc.label_) += csym_sign * csbp(csym_clbl);
      }
      if(mode_ == ResultMode::update) {
        tc.add(cblockid, cbp);
      } else {
        tc.put(cblockid, cbp);
      }
    }
  };
  parallel_work(citr, citr.get_end(), lambda);
}
#endif

inline int
compute_symmetry_scaling_factor(const TensorVec<SymmGroup>& sum_indices,
                                TensorIndex sumid) {
  int ret = 1;
  auto itr = sumid.begin();
  for(auto &sg: sum_indices) {
    auto sz = sg.size();
    Expects(sz > 0);
    std::sort(itr, itr+sz);
    auto fact = factorial(sz);
    int tsize = 1;
    for(int i=1; i<sz; i++) {
      if(itr[i] != itr[i-1]) {
        fact /= factorial(tsize);
        tsize = 0;
      }
      tsize += 1;
    }
    fact /= factorial(tsize);
    ret *= fact;
    itr += sz;
  }
  return ret;
}

template<typename T, typename LabeledTensorType>
inline void
MultOp<T, LabeledTensorType>::execute() {
  using T1 = typename LabeledTensorType::element_type;
  // std::cerr<<__FUNCTION__<<":"<<__LINE__<<": MultOp\n";

  //@todo @fixme MultOp based on nonsymmetrized_iterator cannot work with ResultMode::set
  Expects(mode_ == ResultMode::update);
  LabeledTensor<T1>& lta = rhs1_;
  LabeledTensor<T1>& ltb = rhs2_;
  LabeledTensor<T1>& ltc = lhs_;
  Tensor<T1>& ta = *lta.tensor_;
  Tensor<T1>& tb = *ltb.tensor_;
  Tensor<T1>& tc = *ltc.tensor_;

  TensorLabel sum_labels;
  TensorVec<SymmGroup> sum_indices;
  std::tie(sum_indices, sum_labels) = summation_indices(ltc, lta, ltb);
  auto lambda = [&] (const TensorIndex& cblockid) {
    auto dimc = tc.block_size(cblockid);
    if(!(tc.nonzero(cblockid) && tc.spin_unique(cblockid) && dimc > 0)) {
      // std::cout<<"MultOp. zero block "<<cblockid<<std::endl;
      // std::cout<<"MultOp. "<<cblockid<<" nonzero="<< tc.nonzero(cblockid) <<std::endl;
      // std::cout<<"MultOp. "<<cblockid<<" spin_unique="<< tc.spin_unique(cblockid) <<std::endl;
      // std::cout<<"MultOp. "<<cblockid<<" dimc="<< dimc <<std::endl;
      return;
    }
    // std::cout<<"MultOp. non-zero block"<<std::endl;
    auto sum_itr_first = loop_iterator(slice_indices(sum_indices, sum_labels));
    auto sum_itr_last = sum_itr_first.get_end();
    auto label_map = LabelMap<BlockDim>().update(ltc.label_, cblockid);
    auto cbp = tc.alloc(cblockid);
    cbp() = 0.0;
    for(auto sitr = sum_itr_first; sitr!=sum_itr_last; ++sitr) {
      label_map.update(sum_labels, *sitr);
      auto ablockid = label_map.get_blockid(lta.label_);
      auto bblockid = label_map.get_blockid(ltb.label_);

      // std::cout<<"summation loop. value="<<*sitr<<std::endl<;

      if(!ta.nonzero(ablockid) || !tb.nonzero(bblockid)) {
        continue;
      }
      // ablockid = ta.find_spin_unique_block(ablockid);
      // bblockid = tb.find_spin_unique_block(bblockid);
      auto abp = ta.get(ablockid);
      auto bbp = tb.get(bblockid);

      auto symm_scaling_factor = compute_symmetry_scaling_factor(sum_indices, *sitr);
      auto scale = alpha_ * symm_scaling_factor;
#if 1
      // std::cout<<"doing block-block multiply"<<std::endl;
      cbp(ltc.label_) += alpha_ * abp(lta.label_) * bbp(ltb.label_);
#endif
    }
    auto csbp = tc.alloc(tc.find_unique_block(cblockid));
    csbp() = 0.0;
    auto copy_symm = copy_symmetrizer(ltc, lta, ltb, label_map);
    auto copy_itr = copy_iterator(copy_symm);
    auto copy_itr_last = copy_itr.get_end();
    auto copy_label = TensorLabel(ltc.label_);
    //std::sort(copy_label.begin(), copy_label.end());
    for(auto citr = copy_itr; citr != copy_itr_last; ++citr) {
      auto cperm_label = *citr;
      auto num_inversions = perm_count_inversions(perm_compute(copy_label, cperm_label));
      Sign sign = (num_inversions%2) ? -1 : 1;
      // std::cout<<"doing copy symmetrizer add. sign="<<sign<<std::endl;
      csbp(copy_label) += T(sign) * cbp(cperm_label);
    }
    if(mode_ == ResultMode::update) {
      // std::cout<<"Updating to block id="<<csbp.blockid()<<" value[0]="<<csbp.buf()[0]<<std::endl;
      tc.add(csbp.blockid(), csbp);
    } else {
      tc.put(csbp.blockid(), csbp);
    }
  };
  auto itr = nonsymmetrized_iterator(ltc, lta, ltb);
  parallel_work(itr, itr.get_end(), lambda);
}

}; // namespace tammx

#endif  // TAMMX_OPS_H_
