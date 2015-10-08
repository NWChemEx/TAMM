#ifndef __ctce_iter_group_h__
#define __ctce_iter_group_h__

namespace ctce {

  typedef enum {
    ANTI, TRIG, COPY, TRIG2
  } IterType;

  template <class T>
    class IterGroup {
      private:
        std::vector<T> iters_;
        std::vector< std::vector<Integer> > curr_;
        std::vector<bool> ub_;
        bool empty_;
        int sign_;
        std::vector<Integer> curr_r_;
        std::vector<Integer> curr_o_;
        IterType type_;

      public:
        IterGroup() { empty_ = true; };
        ~IterGroup() {};
        IterGroup(const std::vector<T>& iters, IterType type) 
          : type_(type), empty_(false), sign_(1) {
            for (int i=0; i<iters.size(); i++) {
              if (!iters[i].empty()) iters_.push_back(iters[i]);
            } // skip empty ones
            int n = iters_.size();
            if (n == 0) empty_ = true;
            curr_.resize(n);
            curr_r_.resize(n);
            curr_o_.resize(n);
            ub_.resize(n);
            for (int i=0; i<n; i++) {
              iters_[i].reset();
              ub_[i] = iters_[i].next(curr_[i]);
            }
          }

        const bool& empty() const { return empty_; }
        const int& sign() const { return sign_; }
        const std::vector<Integer>& v_range() const { return curr_r_; }
        const std::vector<Integer>& v_offset() const { return curr_o_; }
        void setType(IterType it) { type_ = it; }

        bool next(std::vector<Integer> & vec) {
          if (empty_) return false; // no summation indices, execute once by checking isEmpty()
          if (ub_[0]==false) return false; // end of iteration
          int n = iters_.size();
          vec.clear();
          for (int i=0; i<n; i++) vec.insert(vec.end(),curr_[i].begin(),curr_[i].end()); // all = curr[0]+curr[1]+...

          // update sign
          if (type_==COPY) { 
            sign_ = 1;
            for (int i=0; i<n; i++) sign_ *= iters_[i].sign();
          }
          if (type_==TRIG2) { // used in ccsd_t.cc
            curr_r_.clear();
            curr_o_.clear();
            for (int i=0; i<n; i++) {
              curr_r_.insert(curr_r_.end(), iters_[i].v_range().begin(), iters_[i].v_range().end());
              curr_o_.insert(curr_o_.end(), iters_[i].v_offset().begin(), iters_[i].v_offset().end());
            }
          }

          // plus one
          int last = curr_.size()-1;
          ub_[last] = iters_[last].next(curr_[last]); // +1

          // compute carry
          for (int i=last; i>0; i--) {
            if (ub_[i]==false) {
              ub_[i-1] = iters_[i-1].next(curr_[i-1]);
                for (int j=i; j<=last; j++) {
                iters_[j].reset();
                ub_[j] = iters_[j].next(curr_[j]);
              }
            }
          }
          return true;
        }

        void reset() {
          if (empty_) return;

          int n = iters_.size();
          for (int i=0; i<n; i++) {
            iters_[i].reset();
            ub_[i] = iters_[i].next(curr_[i]);
          }
        }

        void fix_ids_for_copy(std::vector<Integer>& vec) {
          if (type_!=COPY) return;
          int begin=0, end=0, offset=0;
          for (int i=0; i<iters_.size(); i++) {
            end += iters_[i].size();
            for (int j=begin; j<end; j++) vec[j] += offset;
            begin = end;
            offset += iters_[i].size();
          }
        }
    };

} /* namespace ctce */

#endif /* __ctce_iter_group_h */
