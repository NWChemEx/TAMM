// Copyright 2016 Pacific Northwest National Laboratory

#ifndef TAMMY_GENERATOR_H_
#define TAMMY_GENERATOR_H_

#include "tammy/boundvec.h"
#include "tammy/types.h"

namespace tammy {

template<typename T>
class Generator {
 public:
  virtual ~Generator() {}
  virtual void reset() = 0;
  virtual void reset(const TensorVec<T>& lo) = 0;
  virtual bool has_more() = 0;
  virtual void get(TensorVec<T>& itr) const = 0;
  virtual void next() = 0;
  virtual size_t size() const = 0;
  virtual const TensorVec<T>& lo() const = 0;
  virtual const TensorVec<T>& hi() const = 0;
  virtual const TensorVec<T>& itr_vec() const = 0;

  TensorVec<T> get() const {
    TensorVec<T> ret;
    get(ret);
    return ret;
  }
};  // Generator

template<typename T>
class SimpleGenerator : public Generator<T> {
 public:
  size_t index_;
  T lo_, hi_, step_;
  TensorVec<T> lov_, hiv_;
  mutable TensorVec<T> itr_vec_;
  T it_;
  bool done_;

  SimpleGenerator(size_t index, T lo, T hi)
      : index_{index},
        lo_{lo},
        hi_{hi},
        step_{lo<=hi ? 1 : -1},
        lov_{lo},
        hiv_{hi},
        it_{lo},
        done_{false} {
          reset();
          assert(has_more());
        }

  void reset() override {
    reset(lo_);
  }

  const TensorVec<T>& lo() const override {
    return lov_;
  }

  const TensorVec<T>& hi() const override {
    return hiv_;
  }

  const TensorVec<T>& itr_vec() const override {
    itr_vec_.clear();
    itr_vec_.push_back(it_);
    return itr_vec_;
  }

  virtual size_t size() const override {
    return 1;
  }

  void reset(const TensorVec<T>& lo) override {
    assert(lo.size() == 1);
    reset(lo[0]);
  }

  void reset(T clo) {
    it_ = clo;
    done_ = false;
  }

  bool has_more() override {
    return !done_;
  }

  void get(TensorVec<T>& itr) const override {
    assert(index_ < itr.size());
    itr[index_] = it_;
  }

  void next() override {
    if(it_+step_ == hi_) {
      it_ = lo_;
      done_ = true;
    } else {
      it_ += step_;
    }
  }
};  // SimpleGenerator

template<typename T>
class CartesianGenerator : public Generator<T> {
 public:
  TensorVec<std::unique_ptr<Generator<T>>> itrs_;
  TensorVec<size_t> offs_;
  TensorVec<T> lo_;
  TensorVec<T> hi_;
  mutable TensorVec<T> itr_vec_;
  size_t size_;
  bool done_;

  CartesianGenerator(TensorVec<std::unique_ptr<Generator<T>>>&& itrs)
      : itrs_{std::move(itrs)},
        done_{false} {
          size_ = 0;
          offs_.push_back(0);
          for(const auto& itr: itrs_) {
            size_ += itr->size();
            offs_.push_back(size_);
            lo_.insert_back(itr->lo().begin(), itr->lo().end());
            hi_.insert_back(itr->hi().begin(), itr->hi().end());
          }
          reset();
        }

  void reset() override {
    reset(lo_);
  }

  size_t size() const override {
    return size_;
  }

  const TensorVec<T>& lo() const override {
    return lo_;
  }

  const TensorVec<T>& hi() const override {
    return hi_;
  }

  const TensorVec<T>& itr_vec() const override {
    itr_vec_.clear();
    itr_vec_.resize(size_);
    for(const auto& itr: itrs_) {
      itr->get(vec);
      //const auto& vec = itr->itr_vec();
      //itr_vec_.insert_back(vec.begin(), vec.end());
    }
    return vec;
  }

  void reset(const TensorVec<T>& clo) override {
    assert(clo.size() == size_);
    size_t sz = 0;
    for(size_t i=0; i<itrs_.size(); i++) {
      TensorVec<T> ilo(clo.begin()+offs_[i], clo.begin()+offs_[i+1]);
      itrs_[i]->reset(ilo);
      sz += itrs_[i]->size();
    }
    done_ = false;
  }

  bool has_more() override {
    return !done_;
  }

  void get(TensorVec<T>& itv) const override {
    for(auto& itr: itrs_) {
      itr->get(itv);
    }
  }

  void next() override {
    int i = itrs_.size()-1;
    for(; i>=0; i--) {
      itrs_[i]->next();
      if (itrs_[i]->has_more()) {
        break;
      }
      itrs_[i]->reset();
    }
    if (i<0) {
      done_ = true;
    }
  }
};  // CartesianGenerator

template<typename T>
class TriangleGenerator : public CartesianGenerator<T> {
 public:
  TensorVec<T> clo_;

  TriangleGenerator(TensorVec<std::unique_ptr<Generator<T>>>&& itrs)
      : CartesianGenerator<T>(std::move(itrs)) {
    CartesianGenerator<T>::done_ = false;
    const auto& itrs_ = this->itrs_;
    if(itrs_.size() > 0) {
      size_t sz = itrs_[0]->size();
      for(const auto& itr: itrs_) {
        assert(sz == itr->size());
      }
    }
  }

  void reset(const TensorVec<T>& clo) override {
    assert(clo.size() == this->size_);
    clo_ = clo;
    this->done_ = false;
    auto& itrs = this->itrs_;
    if(itrs.size() > 0) {
      size_t sz = itrs[0]->size();
      for(size_t i = 0; i<itrs.size(); i++) {
        TensorVec<T> rv(clo.begin() + i*sz,
                          clo.begin() + (i+1)*sz);
        assert(rv.size() == itrs[i]->size());
        itrs[i]->reset(rv);
      }
    }
  }

  // @todo @bug std::max cannot be used with the loop has a negative
  // step (i.e., hi is lesser than lo
  void next() override {
    auto& itrs = this->itrs_;
    int i = itrs.size()-1;
    for(; i>=0; i--) {
      itrs[i]->next();
      if (itrs[i]->has_more()) {
        break;
      }
    }
    if (i<0) {
      this->done_ = true;
    } else {
      const auto& ivec = itrs[i]->itr_vec();
      size_t sz = itrs[0]->size();
      for(size_t j=i+1; j<itrs.size(); j++) {
        TensorVec<T> rv(ivec.begin(), ivec.end());
        if(clo_.size() > 0) {
          for(size_t k = 0; k<sz; k++) {
            rv[k] = std::max(rv[k], clo_[j*sz+k]);
          }
        }
        assert(rv.size() == itrs[j]->size());
        itrs[j]->reset(rv);
      }
    }
  }
};  // TriangleGenerator

}  // namespace tammy

#endif // TAMMY_GENERATOR_H_

