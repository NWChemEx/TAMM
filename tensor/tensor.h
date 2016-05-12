#ifndef __ctce_tensor_h__
#define __ctce_tensor_h__

#include "fapi.h"
#include "typesf2c.h"
#include "index.h"
#include "variables.h"
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>
using namespace std;

namespace ctce {

	enum DistType { dist_nwi, dist_nwma, dist_nw};
	enum DimType { dim_ov, dim_n };

  class Tensor {
    private:
      int dim_; /*< dimension of this tensor */
      int sign_; /*< sign of this tensor: 1 or -1 */
      std::vector<IndexName> mem_pos_;	/*< memory position of the indices */
      std::vector<int> tab_; /*< map(Index,int): (p4,0)(p5,1)(p6,2)(h1,3)(h2,4)(h3,5) */
      TensorType type_; /*< type of this tensor: F_tensor, T_tensor, etc. */
      std::vector<int> ext_sym_group_; /*< external symmetry group of this tensor */

      /* initial setting, will not change */
      std::vector<Index> ids_; /*< indices of the tensor, actual data */

      /* name, value, value_r of the indices to avoid accessing ids_ every time */
      std::vector<IndexName> name_; /*< indices name of this tensor: (p1,p2,p3,p4) */
      std::vector<Integer> value_; /*< indices value of this tensor: (0,1,2,3) */
      std::vector<Integer> value_r_; /*< indices restricted value of this tensor */

      /* replicates for sorting */
      std::vector<Index> _ids_; /*< indices of this tensor after sorting */
      std::vector<IndexName> _name_; /*< indices name of this tensor after sorting */
      std::vector<Integer> _value_; /*< indices value of this tensor after sorting */
      std::vector<Integer> _value_r_; /*< indices restricted value of this tensor after sorting*/

      std::vector<int> pos1; /*< pos1 = (0 1 2 3 ...) */
      std::vector<int> pos2; /*< position of the indices after sorting, use to find sign by comparing it to pos1 */
      std::vector<Integer> sort_ids_;
      std::vector<Integer> sort_ids_v_;
      std::vector<Integer> perm_;

    public:
			DistType dist_type_;
			DimType dim_type_;

      /**
       * Constructor
       */
      Tensor() {};
      /**
       * Destructor
       */
      ~Tensor() {};

      /* bool get_ma; /\*< if true, will call get_hash_block_ma_ *\/ */
      /* bool get_i; /\*< if true, will call get_hash_block_i *\/ */

      /**
       * Constructor. Assign data to this tensor
       * @param[in] n Dimension of this tensor
       * @param[in] ids Indices of this tensor
       * @param[in] type Type of this tensor
       */
	Tensor(const int& n, Index ids[], TensorType type, DistType dist_type, DimType dim_type)
        : dim_(n),
        type_(type),
        sign_(1),
				dist_type_(dist_type),
				dim_type_(dim_type) {
          ids_.resize(n);
          name_.resize(n);
          value_.resize(n);
          value_r_.resize(n);
          pos1.resize(n);
          pos2.resize(n);
          sort_ids_.resize(n);
          sort_ids_v_.resize(n);
          perm_.resize(n);
          ext_sym_group_.resize(n);
          tab_.resize(IndexNum);
          for(int i=0; i<IndexNum; i++) {
            tab_[i]=-1;
          }
          for(int i=0; i<n; i++) {
            ids_[i]=ids[i];
            name_[i] = ids[i].name();
            value_[i] = ids[i].value();
            value_r_[i] = ids[i].value_r();
            tab_[ids[i].name()] = i;
            ext_sym_group_[i] = ids[i].ext_sym_group();
            pos1[i]=i;
          }
          /* replicates */
          _ids_ = ids_;
          _name_ = name_;
          _value_ = value_;
          _value_r_ = value_r_;
          /* get_ma = false; */
          /* get_i = false; */
        }

      /**
       * Get the dimension of this tensor
       * @return dim as a int
       */
      inline const int& dim() const { return dim_; }

      /**
       * Get the sign of the tensor
       * @return sign as a int
       */
      inline const int& sign() const { return sign_; }

      /**
       * Get the indices of the tensor
       * @return ids as a vector of Index
       */
      inline const std::vector<Index>& ids() const { return ids_; }

      /**
       * Get the indices name of the tensor
       * @return name as a vector of IndexName
       */
      inline const std::vector<IndexName>& name() const { return name_; }

      /**
       * Get the indices value of the tensor
       * @return value as a vector of Integer
       */
      inline std::vector<Integer>& value() { return value_; }

      /**
       * Get the indices restricted value of the tensor
       * @return value_r as a vector of Integer
       */
      inline std::vector<Integer>& value_r() { return value_r_; }

      /**
       * Get the indices name of the tensor after sorting
       * @return name as a vector of IndexName
       */
      inline const std::vector<IndexName>& _name() const { return _name_; }

      /**
       * Get the indices value of the tensor after sorting
       * @return value as a vector of Integer
       */
      inline std::vector<Integer>& _value() { return _value_; }

      /**
       * Get the indices restricted value of the tensor after sorting
       * @return value as a vector of Integer
       */
      inline std::vector<Integer>& _value_r() { return _value_r_; }

      /**
       * Get the type of the tensor
       * @return type as TensorType
       */
      inline const TensorType& type() const { return type_; }

      /**
       * Get the external symmetry group of the tensor
       * @return ext_sym_group as a vector of int
       */
      inline const std::vector<int>& ext_sym_group() const { return ext_sym_group_; }

      /**
       * Get the indices name of in the memory position order
       * @return getMemPosName as a vector of IndexName
       */
      inline const std::vector<IndexName>& getMemPosName() const { return mem_pos_; }

      /**
       * Check if this tensor is an intermediate: iF_tensor, iV_tensor, etc.
       * @return bool value
       */
      inline bool isIntermediate() {
        if ((type_==iV_tensor)||(type_==iT_tensor)||(type_==iF_tensor)||
            (type_==iVT_tensor)||(type_==iVF_tensor)||(type_==iTF_tensor))
          return true;
        return false;
      }

      // the following 4 methods need global variables, method body in tensor.cc
      /**
       * Get the corresponding irrep value from the global variables
       * @return irrep as Integer
       */
      const Integer& irrep();

      /**
       * Get data by get_hash_block_xx and store in buf, this function is for t_mult
       * @param[in] d_a
       * @param[in] *buf
       * @param[in] size size of the buf
       * @param[in] d_a_offset
       */
      void get(Integer d_a, double *buf, Integer size, Integer d_a_offset);

      /**
       * Get data by get_hash_block_xx and store in buf, this function is for t_assign
       * @param[in] d_a
       * @param[in] *buf
       * @param[in] size size of the buf
       * @param[in] d_a_offset
       */
      //void get2(Integer d_a, double *buf, Integer size, Integer d_a_offset);

			void get2(Integer d_a, std::vector<Integer> &pvalue_r, double *buf, Integer size, Integer d_a_offset);

      /**
       * Generate restricted value from value by calling tce_restricted2/4
       */
      void gen_restricted();

      void gen_restricted(const std::vector<Integer> &value,
													std::vector<Integer> &pvalue_r);

      /**
       * Set the memory position for the indices
       * @param[in] mp memory position of the indices as a vector of IndexName
       */
      inline void setMemPos(const std::vector<IndexName>& mp) { mem_pos_ = mp; }

      /**
       * Set the value of the indices by name
       * @param[in] name Name of the index to set
       * @param[in] value Value of the index to set
       */
      inline void setValueByName(const IndexName& name, const Integer& value) {
        int pos = tab_[name];
        assert(pos>=0 && pos <=ids_.size());
        ids_[pos].setValue(value);
        value_[pos]=value;
      }

      /**
       * Set the restricted value of indices
       * @param val restricted value as a vector of Integer
       */
      inline void setValueR(const std::vector<Integer>& val) {
        assert(ids_.size()==val.size());
        for (int i=0; i<ids_.size(); i++) {
          ids_[i].setValueR(val[i]);
          value_r_[i]=val[i];
        }
      }

      /**
       * Sorting method, first sort the indices by value, and then by external symmetry group
       * The initial data remain unchanged, only modify _ids_, _name_, _value_, _value_r_
       */
      inline void sortByValueThenExtSymGroup() {
        int n = ids_.size();
        _ids_ = ids_;
        std::sort(_ids_.begin(),_ids_.end(),compareValue);
        std::sort(_ids_.begin(),_ids_.end(),compareExtSymGroup);
        for (int i=0; i<n; i++) pos2[i]=tab_[_ids_[i].name()];
        sign_ = countParitySign<int>(pos1,pos2);
        for (int i=0; i<n; i++) {
          _name_[i] = _ids_[i].name();
          _value_[i] = _ids_[i].value();
          _value_r_[i] = _ids_[i].value_r();
        }
      }

      /**
      * Re-order the indices according to a input vector of Integer
      * For example, (0,1,3,2) will swap the 3rd and 4th indices position
      * @param[in] order new position of the indices as a vector of Integer
      */
      inline void orderIds(const std::vector<Integer>& order) {
        int n = ids_.size();
        for (int i=0; i<n; i++) {
          assert(order[i]>=0 && order[i]<n);
          _ids_[i]=ids_[order[i]];
        }
        for (int i=0; i<n; i++) {
          _name_[i] = _ids_[i].name();
          _value_[i] = _ids_[i].value();
          _value_r_[i] = _ids_[i].value_r();
        }
      }

      /**
      * Get the value of the indices in memory position order
      * @return memory position value as vector of Integer
      */
      inline std::vector<Integer>& getMemPosVal() {
        for (int i=0; i<dim_; i++) {
          int pos = tab_[mem_pos_[i]];
          sort_ids_v_[i]=ids_[pos].value();
        }
        return sort_ids_v_;
      }

      /**
      * Get the position of the indices in memory from the position of sorted indices
      * @return sort_ids vector of Integer indicates the position
      */
      inline std::vector<Integer>& sort_ids() {
        for (int i=0; i<dim_; i++) {
          sort_ids_[i] = std::find(_name_.begin(), _name_.end(), mem_pos_[i]) - _name_.begin() + 1;
        }
        return sort_ids_;
      }

      /**
      * Get the position of the sorted indices from the position in the memory
      * @return perm vector of Integer indicates the position
      */
      inline std::vector<Integer>& perm() {
        for (int i=0; i<dim_; i++) {
          perm_[i] = std::find(mem_pos_.begin(), mem_pos_.end(), _name_[i]) - mem_pos_.begin() + 1;
        }
        return perm_;
      }

  }; // Tensor

  extern "C" {

    /**
    * Function that can create a 2-d tensor
    * @param[in] n1 first index name
    * @param[in] n2 second index name
    * @param[in] e1 first index symmetry group, should be 0
    * @param[in] e2 second index symmetry group, should be 1
    * @param[in] type type of the tensor
    */
    Tensor Tensor2(IndexName n1, IndexName n2, int e1, int e2, TensorType type, DistType dt=dist_nw, DimType dm=dim_ov);

    /**
    * Function that can create a 4-d tensor
    * @param[in] n1 first index name
    * @param[in] n2 second index name
    * @param[in] n3 third index name
    * @param[in] n4 fourth index name
    * @param[in] e1 first index symmetry group, should start with 0
    * @param[in] e2 second index symmetry group
    * @param[in] e3 first index symmetry group
    * @param[in] e4 second index symmetry group
    * @param[in] type type of the tensor
    */
    Tensor Tensor4(IndexName n1, IndexName n2, IndexName n3, IndexName n4,
									 int e1, int e2, int e3, int e4, TensorType type, DistType dt=dist_nw, DimType dm=dim_ov);

    /**
    * Function that can create a 6-d tensor
    * @param[in] n1 first index name
    * @param[in] n2 second index name
    * @param[in] n3 third index name
    * @param[in] n4 fourth index name
    * @param[in] n5 fifth index name
    * @param[in] n6 sixth index name
    * @param[in] e1 first index symmetry group, should start with 0
    * @param[in] e2 second index symmetry group
    * @param[in] e3 first index symmetry group
    * @param[in] e4 second index symmetry group
    * @param[in] e5 fifth index symmetry group
    * @param[in] e6 sixth index symmetry group
    * @param[in] type type of the tensor
    */
    Tensor Tensor6(IndexName n1, IndexName n2, IndexName n3, IndexName n4, IndexName n5, IndexName n6,
									 int e1, int e2, int e3, int e4, int e5, int e6, TensorType type, DistType dt=dist_nw, DimType dm=dim_ov);

  };

} /* namespace ctce */

#endif /* __ctce_tensor_h__ */
