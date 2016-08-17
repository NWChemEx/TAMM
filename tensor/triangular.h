#ifndef __ctce_triangular_h__
#define __ctce_triangular_h__

#include "index.h"
#include "variables.h"

namespace ctce {

  class triangular {

    private:
      bool empty_; /*< indicate if this triangular loops is empty */
      std::vector<size_t> lb; /*< lower bound of the loops */
      std::vector<size_t> ub; /*< upper bound of the loops */
      std::vector<size_t> curr; /*< current value of the loops */
      std::vector<size_t> curr_lb; /*< current lower bound of the loops */
      
      // these are only use in ccsd(t) fusion TRIG2
      std::vector<size_t> r;
      std::vector<size_t> o;
      std::vector<size_t> curr_r;
      std::vector<size_t> curr_o;

      std::vector<bool> dirty; /*< dirty indicate if carry occur */

      /**
      * Update the r and o in ccsd(t) fusion part
      */
      void updateRO() {
        if(curr.size()==0) return; 
        if (curr[0]>ub[0]) return; // no need to compute
        Integer *int_mb = Variables::int_mb();
        size_t k_range = Variables::k_range();
        size_t k_evl_sorted = Variables::k_evl_sorted();
        size_t k_offset = Variables::k_offset();
        for (int i=curr.size()-1; i>=0; i--) {
          if (dirty[i]) {
            r[i] = int_mb[k_range + curr[i] - 1];
            o[i] = k_evl_sorted + int_mb[k_offset + curr[i] - 1] - 1;
            dirty[i]=false;
          }
          else return;
        }
      }

    public:
      /**
      * Constructor
      */
      triangular() {};

      /**
      * Destructor
      */
      ~triangular() {};

      /**
      * Constructor. Assign indices name for this triangular loops
      * @param[in] ids Name of the indices for this triangular loops
      */
      explicit triangular(const std::vector<IndexName>& ids) {
        int d = ids.size();
        lb.resize(d);
        ub.resize(d);
        r.resize(d);
        o.resize(d);
        dirty.resize(d);
        if (d==0) empty_ = true;
        else {
          empty_ = false;
          int range = Table::rangeOf(ids[0]); // name
          for (int i=0; i<d; i++) {
            if (range==TO) {
              lb[i] = 1;
              ub[i] = Variables::noab();
            }
            else if (range==TV) {
              lb[i] = Variables::noab()+1;
              ub[i] = Variables::noab()+Variables::nvab();
            }
            dirty[i]=true;
          }
          curr = lb;
          curr_lb = lb;
        }
        updateRO();
        curr_r = r;
        curr_o = o;
      }

      /**
      * Get the current iterate value, store to vec, and iterate next
      * @param[in] vec current iterate value
      * @return true the value is valid, return false when end of the iterator
      */
      inline bool next(std::vector<size_t>& vec) {
        if (empty_) return false;
        if (curr[0]>ub[0]) return false;
        vec = curr;
        curr_r = r;
        curr_o = o;
        // plus 1
        curr[curr.size()-1]++;
        dirty[curr.size()-1]=true;
        for (int i=curr.size()-1; i>0; i--) { // compute carry
          if (curr[i]>ub[i]) {
            // carry 1
            curr[i-1]++;
            dirty[i-1]=true;
            // set lowerbound
            for (int j=curr.size()-1; j>=i; j--) {
              curr_lb[j]=curr[i-1];
              curr[j]=curr_lb[j];
            }
          }
        }
        updateRO();
        return true;
      }

      /**
      * Reset the iterator
      */
      inline void reset() { 
        curr = lb; 
        curr_lb = lb; 
        for (int i=0; i<curr.size(); i++) dirty[i]=true;
        updateRO();
      }
      
      /**
      * Check if the iterator is empty, which means no indices name assigned
      */
      inline const bool& empty() const { return empty_; }

      /** 
      * No used, just for passing compilation for IterGroup class
      */ 
      inline const int sign() const { return 1; }

      /**
      * Return current r value
      */
      inline const std::vector<size_t>& v_range() const { return curr_r; }
      
      /**
      * Return current o value
      */
      inline const std::vector<size_t>& v_offset() const { return curr_o; }
  };

} /* namespace ctce */

#endif /* __ctce_triangular_h__ */
