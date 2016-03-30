#ifndef __ctce_antisymm_h__
#define __ctce_antisymm_h__

#include "variables.h"

namespace ctce {

  /**
   * Triangular loop class
   */
  class tloop {
    private:
      int lb, ub;
      std::vector<Integer> indices;
      int nloops;
      bool first_time;
      std::vector<int> loops;
      inline void complete() {
        if(loops.size()==0) loops.push_back(lb);
        while(loops.size() < nloops) loops.push_back(loops.back()+1);
      }
      inline int nextpos(int pos) {
        assert(pos<indices.size());
        int ret;
        for(ret=pos+1; ret<indices.size() && indices[ret]==indices[pos]; ret++);
        return ret;
      }
    public:
      /**
       *Constructor
       */
      tloop() {};

      /**
       * Destructor
       */
      ~tloop() {};

      /**
       * Constructor
       *
       */
      tloop(const std::vector<Integer> &_indices, const int& _nloops)
        : indices(_indices), 
        nloops(_nloops), 
        first_time(true), 
        lb(0) {
          ub = indices.size();
          complete();
        }
      inline const std::vector<int>& vec() const { return loops; }
      inline bool nextIter() {
        if(first_time == true) {
          first_time = false;
          return true;
        }
        while(loops.size()>0 && (indices.size()-nextpos(loops.back()) <= nloops-loops.size())) loops.pop_back();
        if(loops.size()==0) return false;
        loops.back() = nextpos(loops.back());
        complete();
        return true;
      }
  };

  /**
   * Antisymm iterator. Iterate input (3 2)(4) as (2 3 4),(2 4 3),(3,4,2) where (a<b)(c) 
   */
  class antisymm {
    private:
      tloop tl0; /*< initial setting of tloop */
      tloop tl1; /*< relicate for iterating */
      std::vector<Integer> slist; /*< current value */
      std::vector<Integer> s1; /*< current value from A tensor */
      std::vector<Integer> s2; /*< current value from B tensor */
      bool empty_; /*< check if antisymm is empty */
    public:
      /**
       * Constructor
       */
      antisymm() {};

      /** 
       * Destructor
       */ 
      ~antisymm() {};

      /**
       * Constructor
       * @param[in] name IndexName of the iterator
       * @param[in] s1 number of anti-symmetry group from A tensor
       * @param[in] s2 number of anti-symmetry group from B tensor
       */
      antisymm(const std::vector<IndexName>& name, int s1, int s2);

      /**
       * Check if this iterator is empty
       */
      inline const bool& empty() const { return empty_; }

      /**
       * Reset this anti-symmetry iterator
       */
      inline void reset() { tl1 = tl0; }

      /**
       * Get current value of iterator.
       * @param[in] vec get current value and store it in vec
       * @return return false if end of iteration
       */
      inline bool next(std::vector<Integer> & vec); /*< enumerate next permuation for computation, store in &vec */

      /* following 3 no use, just for passing compilation for iterGroup */
      inline const int sign() const { return 1; }
      inline const std::vector<Integer>& v_range() { };
      inline const std::vector<Integer>& v_offset() { };
  };

  inline antisymm::antisymm(const std::vector<IndexName>& name, int n1, int n2) {
    int n = name.size();
    slist.resize(n);
    if (n==0) empty_=true;
    else {
      empty_=false;
      const std::vector<Integer>& vtab = Table::value(); /* get value from table */
      for (int i=0; i<n; i++) slist[i] = vtab[name[i]];
      sort(slist.begin(), slist.end());
    }
    tl0 = tloop(slist, n1);
    tl1 = tl0;
  }

  inline bool antisymm::next(std::vector<Integer> & vec) {
    if (!tl1.nextIter()) return false;
    const std::vector<int>& itr = tl1.vec();
    s1.clear();
    s2.clear();
    int p=0;
    for(int i=0; i<itr.size(); i++) {
      while (p<itr[i]) {
        s2.push_back(slist[p]);
        p++;
      }
      s1.push_back(slist[itr[i]]);
      p = itr[i] + 1;
    }
    while(p<slist.size()) {
      s2.push_back(slist[p]);
      p++;
    }
    vec = s1;
    vec.insert(vec.end(), s2.begin(), s2.end());
    return true;
  }

} /* namespace ctce */

#endif /*__ctce_antisymm_h__*/
