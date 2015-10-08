#ifndef __ctce_expr_h__
#define __ctce_expr_h__

#include "typesf2c.h"
#include "index.h"
#include <vector>
#include <cassert>
#include <iostream>
#include "tensor.h"
#include "triangular.h"
#include "iterGroup.h"

namespace ctce {

  /**
   * Assigment template. tC += coef * tA
   */
  class Assignment {
    private:
      Tensor tC_; /*< lhs tensor */
      Tensor tA_; /*< rhs tensor */
      double coef_; /*< coefficient */
      IterGroup<triangular> out_itr_; /*< outer loop iterator */
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
      Assignment(const Tensor& tC, const Tensor& tA, double coef)
        : tC_(tC), tA_(tA), coef_(coef) {
          init();
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
