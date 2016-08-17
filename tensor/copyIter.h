#ifndef __ctce_copy_iter_h__
#define __ctce_copy_iter_h__

#include "typesf2c.h"
#include <vector>
#include <cstdlib>

namespace ctce {

  /* a copy iterator that iterates all possible indices for tce_sort_acc and tce_add_hash_block */
  class CopyIter {

    private:
      std::vector< std::vector<size_t> > vec_; /*< vec_ store all the possible permutation */
      std::vector<int> sign_; /*< sign store the sign for each permutation */
      bool empty_; /*< true if this copy iterator is empty */
      int size_; /*< size of the iterator */
      int curr_pos_; /*< current position of the iterator */
      int curr_sign_; /*< current sign of the iterator */
    public:
      /**
       * Constructor
       */
      CopyIter() {};

      /**
       * Destructor
       */
      ~CopyIter() {};

      /**
       * Constructor. Assign the vec and sign to this iterator.
       * Current vec and sign are hardcoded in dummy.cc
       */
      CopyIter(const std::vector< std::vector<size_t> >& v, const std::vector<int>& s)
        : vec_(v), 
        sign_(s), 
        curr_pos_(0), 
        curr_sign_(1), 
        size_(s.size()) {
          if (vec_.size()==0) empty_ = true;
          else empty_ = false;
        };

      /**
       * Check if this iterator is empty
       */
      inline const bool& empty() const { return empty_; }

      /**
       * Return size of this iterator, this method is used in fix_ids_for_cp function in iterGroup.h
       */
      inline const int& size() const { return size_; } // for fix_ids_for_cp

      /** 
       * Get sign of current iteration
       */
      inline const int& sign() const { return curr_sign_; }

      /**
       * Reset this iterator
       */
      inline void reset() { curr_pos_=0; curr_sign_ = sign_[0]; }

      /**
       * Get vec of current iteration
       * return false if end of iteration
       */ 
      inline bool next(std::vector<size_t>& vec) {
        if (curr_pos_ == vec_.size()) return false; // end of iteration
        vec = vec_[curr_pos_];
        curr_sign_ = sign_[curr_pos_];
        curr_pos_++;
        return true;
      };

      /* following 2 methods no use, just to pass IterGroup compilation */
      inline const std::vector<size_t>& v_range() {}; // not used
      inline const std::vector<size_t>& v_offset() {}; // not used

  };

} /* namespace ctce */

#endif /* __ctce_copy_iter_h__ */

