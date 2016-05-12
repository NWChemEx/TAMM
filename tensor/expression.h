#ifndef __ctce_expr_h__
#define __ctce_expr_h__

#include "typesf2c.h"
#include "index.h"
#include <vector>
#include <set>
#include <cassert>
#include <iostream>
#include "tensor.h"
#include "triangular.h"
#include "iterGroup.h"

namespace ctce {

	static inline std::vector<IndexName> ivec(IndexName i1) {
		return std::vector<IndexName>(i1);
	}
	static inline std::vector<IndexName> ivec(IndexName i1, IndexName i2) {
		std::vector<IndexName> ret = ivec(i1);
		ret.push_back(i2);
		return ret;
	}
	static inline std::vector<IndexName> ivec(IndexName i1, IndexName i2, IndexName i3) {
		std::vector<IndexName> ret = ivec(i1,i2);
		ret.push_back(i3);
		return ret;
	}
	static inline std::vector<IndexName> ivec(IndexName i1, IndexName i2, IndexName i3,
															IndexName i4) {
		std::vector<IndexName> ret = ivec(i1,i2,i3);
		ret.push_back(i4);
		return ret;
	}
	static inline bool is_permutation(const std::vector<IndexName>& ids) {
		std::set<IndexName> sids;
		for(int i=0; i<ids.size(); i++) {
			std::cout<<"is_perm. id="<<ids[i]<<endl;
			sids.insert(ids[i]);
		}//ids.begin(), ids.end());
		return sids.size() == ids.size();
	}
	static inline bool is_permutation(const std::vector<IndexName>& ids1, const std::vector<IndexName>& ids2) {
		std::set<IndexName> sids1;
		std::set<IndexName> sids2;
		for(int i=0; i<ids1.size(); i++) {
			sids1.insert(ids1[i]);
		}
		for(int i=0; i<ids2.size(); i++) {
			sids2.insert(ids2[i]);
		}

		if(ids1.size() != sids1.size()) return false;
		if(ids2.size() != sids2.size()) return false;
		for (int i=0; i<ids1.size(); i++) {
			if(sids2.find(ids1[i]) == sids2.end())
				return false;
		}
		return true;
	}

  /**
   * Assigment template. tC += coef * tA
   */
  class Assignment {
    private:
      Tensor tC_; /*< lhs tensor */
      Tensor tA_; /*< rhs tensor */
      double coef_; /*< coefficient */
      IterGroup<triangular> out_itr_; /*< outer loop iterator */
			std::vector<IndexName> cids_;
			std::vector<IndexName> aids_;
			std::vector<int> perm_;
      void init();

    public:
      /**
       * Constructor
       */
      Assignment() {};

      /**
       * Destructor
       */
      ~Assignment() {};

      /**
       * Constructor. Assign tC += coef * tA
       * @param[in] tC left hand side tensor
       * @param[in] tA right hand side tensor
       * @param[in] coef coefficient. most of time it is 1 or -1.
       */
	Assignment(const Tensor& tC, const Tensor& tA, double coef,
						 const std::vector<IndexName>& cids,
						 const std::vector<IndexName>& aids)
		: tC_(tC), tA_(tA), coef_(coef), cids_(cids), aids_(aids) {
          init();
					assert(is_permutation(cids_));
					assert(is_permutation(aids_));
					assert(is_permutation(cids_, aids_));
					for(unsigned i=0; i<aids_.size(); i++) {
						perm_.push_back(std::find(cids_.begin(), cids_.end(), aids_[i])
														- cids_.begin());
					}
        }

      /**
       * Get lhs tensor tC
       */
      Tensor& tC() { return tC_; }

      /**
       * Get rhs tensor tA
       */
      Tensor& tA() { return tA_; }

      /**
       * Get coefficient
       */
      double coef() { return coef_; }

      /**
       * Get outer loop iterator
       */
      IterGroup<triangular>& out_itr() { return out_itr_; }
  };


  /**
   * Multiplication template. tC += coef * tA * tB;
   */
  class Multiplication {
    private:
      Tensor tC_; /*< left hand side tensor */
      Tensor tA_; /*< right hand side tensor 1 */
      Tensor tB_; /*< right hand side tensor 2 */
      double coef_; /*< coefficient */

      std::vector<IndexName> sum_ids_; /*< summation indices of the contraction */
      IterGroup<triangular> out_itr_; /*< outer loop iterator */
      IterGroup<triangular> sum_itr_; /*< summation loop iterator */
      IterGroup<CopyIter> cp_itr_; /*< copy iterator, use to do tC add hash block */

      void genMemPos(); /*< generate memory position according to the indices order of the expression */
      void genTrigItr(IterGroup<triangular>& itr, const std::vector<int>& gp,
          const std::vector<IndexName>& name); /*< generate triangular loops given IndexName and symmetry group */
      void genSumGroup(); /*< generate summation symmetry group and set sum_itr */
      void genCopyGroup(); /*< genertate copy group and set cp_itr */
      void genOutGroup(); /*< generate outer loop group and set out_itr */

    public:

      /**
       * Constructor
       */
      Multiplication() {};

      /**
       * Destructor
       */
      ~Multiplication() {};

      /**
       * Constructor. Assign tC += coef * tA * tB.
       * @param[in] tC left hand side tensor
       * @param[in] tA right hand side tensor 1
       * @param[in] tB right hand side tensor 2
       * @param[in] coef coefficient
       */
      Multiplication(const Tensor& tC, const Tensor& tA, const Tensor& tB, double coef)
        : tC_(tC), tA_(tA), tB_(tB), coef_(coef) {
          genMemPos();
          genSumGroup();
          genOutGroup(); 
        }

      /**
       * Get left hand side tensor tC
       */
      Tensor& tC() { return tC_; }

      /**
       * Get right hand side tensor tA
       */
      Tensor& tA() { return tA_; }

      /**
       * Get right hand side tensor tB
       */
      Tensor& tB() { return tB_; }

      /**
       * Get coefficient
       */
      const double& coef() { return coef_; }

      /**
       * Get summation indices
       */
      std::vector<IndexName>& sum_ids() { return sum_ids_; }

      /**
       * Get outer loop iterator
       */
      IterGroup<triangular>& out_itr() { return out_itr_; }

      /**
       * Get summation indices iterator
       */
      IterGroup<triangular>& sum_itr() { return sum_itr_; }

      /**
       * Get copy iterator
       */
      IterGroup<CopyIter>& cp_itr() { return cp_itr_; }

      /**
       * Manually set out_itr with given symmetry group
       */
      void setOutItr(const std::vector<int>& gp);

      /**
       * Manually set sum_itr with given symmetry group
       */
      void setSumItr(const std::vector<int>& gp);

      /**
       * Manually set cp_itr with given copy group 
       */
      void setCopyItr(const std::vector<int>& gp);
  };

};

#endif
