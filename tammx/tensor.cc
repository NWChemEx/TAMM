#include "tammx/types.h"
#include "tammx/tensor.h"
#include "tammx/labeled-tensor.h"


LabeledTensor
Tensor::operator () (const TensorLabel& label) {
  Expects(label.size() == rank());
  return LabeledTensor{this, label};
}

LabeledTensor
Tensor::operator () () {
  TensorLabel label;
  for(int i=0; i<rank(); i++) {
    label.push_back({i, flindices_[i]});
  }
  return operator ()(label);
}

LabeledTensor
Tensor::operator () (IndexLabel ilbl0) {
  return operator () (TensorLabel{ilbl0});
}

LabeledTensor
Tensor::operator () (IndexLabel ilbl0, IndexLabel ilbl1) {
  return operator () (TensorLabel{ilbl0, ilbl1});
}

LabeledTensor
Tensor::operator () (IndexLabel ilbl0, IndexLabel ilbl1, IndexLabel ilbl2) {
  return operator () (TensorLabel{ilbl0, ilbl1, ilbl2});
}
  
LabeledTensor
Tensor::operator () (IndexLabel ilbl0, IndexLabel ilbl1, IndexLabel ilbl2, IndexLabel ilbl3) {
  return operator () (TensorLabel{ilbl0, ilbl1, ilbl2, ilbl3});
}


void
Tensor::allocate() {
  if (distribution_ == Distribution::tce_nwma || distribution_ == Distribution::tce_nw) {
    ProductIterator<TriangleLoop> pdt =  loop_iterator(indices_);
    auto last = pdt.get_end();
    int length = 0;
    int x=0;
    for(auto itr = pdt; itr != last; ++itr) {
      //std::cout<<x++<<std::endl;
      //std::cout<<"allocate. itr="<<*itr<<std::endl;
      if(nonzero(*itr)) {
        length += 1;
      }
    }
    //FIXME:Handle Scalar
    // if (indices_.size() == 0 && length == 0) length = 1;

    tce_hash_ = new TCE::Int [2 * length + 1];
    tce_hash_[0] = length;
    //start over
    pdt =  loop_iterator(indices_);
    last = pdt.get_end();
    TCE::Int size = 0;
    int addr = 1;
    for(auto itr = pdt; itr != last; ++itr) {
      auto blockid = *itr;
      if(nonzero(blockid)) {
        //std::cout<<"allocate. set keys. itr="<<*itr<<std::endl;
        tce_hash_[addr] = TCE::compute_tce_key(flindices(), blockid);
        tce_hash_[length + addr] = size;
        size += block_size(blockid);
        addr += 1;
      }
    }
    size = (size == 0) ? 1 : size;
    if (distribution_ == Distribution::tce_nw) {
#if 0
      tce_ga_ = tamm::gmem::create(tamm::gmem::Double, size, std::string{"noname1"});
      tamm::gmem::zero(tce_ga_);
#else
      assert(0);
#endif
    }
    else {
      tce_data_buf_ = new uint8_t [size * element_size()];
      typed_zeroout(element_type_, tce_data_buf_, size);
    }
  }
  else {
    assert(0); // implement
  }
  constructed_ = true;
  policy_ = AllocationPolicy::create;
}
