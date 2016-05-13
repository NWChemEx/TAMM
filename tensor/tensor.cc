#include "tensor.h"
#include <iostream>
using namespace std;

namespace ctce {


  const Integer& Tensor::irrep() {
    if ((type_==iV_tensor)||(type_==V_tensor)) return Variables::irrep_v();
    if ((type_==iT_tensor)||(type_==T_tensor)) return Variables::irrep_t();
    if ((type_==iF_tensor)||(type_==F_tensor)) return Variables::irrep_f();
  }

  void Tensor::gen_restricted() {
    std::vector<Integer> temp;
    temp.resize(dim_);
    if (dim_==2)  
      tce_restricted_2_(&value_[0],&value_[1],&temp[0],&temp[1]);
    if (dim_==4)
      tce_restricted_4_(&value_[0],&value_[1],&value_[2],&value_[3],
          &temp[0],&temp[1],&temp[2],&temp[3]);
    for (int i=0; i<dim_; i++) {
      ids_[i].setValueR(temp[i]);
      value_r_[i] = temp[i];
    }
  }

  void Tensor::gen_restricted(const std::vector<Integer>& value,
			      std::vector<Integer> &pvalue_r) {
    std::vector<Integer> temp;
    temp.resize(dim_);
    if (dim_==2)  
      tce_restricted_2_(&value[0],&value[1],&temp[0],&temp[1]);
    if (dim_==4)
      tce_restricted_4_(&value[0],&value[1],&value[2],&value[3],
          &temp[0],&temp[1],&temp[2],&temp[3]);
    pvalue_r.clear();
    for (int i=0; i<dim_; i++) {
      //ids_[i].setValueR(temp[i]);
      //value_r_[i] = temp[i];
      pvalue_r.push_back(temp[i]);
    }
  }

  void Tensor::get(Integer d_a, double *buf, Integer size, Integer d_a_offset) {
    std::vector<Integer>& is = _value_r_;
    std::vector<IndexName>& ns = _name_;
    Integer key = 0, offset = 1;
    Integer noab = Variables::noab();
    Integer nvab = Variables::nvab();
    Integer *int_mb = Variables::int_mb();
    int n = is.size(); // = dim_;
#if 0
    if (get_i && !get_ma) { // different key for get_i
      for (int i=n-1; i>=0; i--) {
        key += (is[i]-1) * offset;
        offset *= noab + nvab;
      }
    }
    else { // normal key
      for (int i=n-1; i>=0; i--) {
        bool check = (Table::rangeOf(ns[i])==TO);
        if (check) key += (is[i]-1)*offset;
        else key += (is[i]-noab-1)*offset; // TV
        offset *= (check)?noab:nvab;
      }
    }
    if (n==2) {
      if (get_ma) {
        double *dbl_mb = Variables::dbl_mb();
	assert(dist_type_ == dist_nwma);
        get_hash_block_ma_(&dbl_mb[d_a], buf, &size, &int_mb[d_a_offset], &key);
      }
      else {
	assert(dist_type_ == dist_nw);
        get_hash_block_(&d_a, buf, &size, &int_mb[d_a_offset], &key);
      }
    }
    else if (n==4) {
      if (get_i) {
	//assert(dist_type_ == dist_nwi);
	if(dist_type_ == dist_nw) {
	  get_hash_block_(&d_a, buf, &size, &int_mb[d_a_offset], &key);
	}
	else {
        get_hash_block_i_(&d_a, buf, &size, &int_mb[d_a_offset], &key,
            &is[3], &is[2], &is[1], &is[0]); // special case
	}
      }
      else {
	assert(dist_type_ == dist_nw);
        get_hash_block_(&d_a, buf, &size, &int_mb[d_a_offset], &key);
      }
    }
#else
    if(dim_type_ == dim_n) {
      for (int i=n-1; i>=0; i--) {
        key += (is[i]-1) * offset;
        offset *= noab + nvab;
      }
    }
    else if(dim_type_ == dim_ov) {
      for (int i=n-1; i>=0; i--) {
        bool check = (Table::rangeOf(ns[i])==TO);
        if (check) key += (is[i]-1)*offset;
        else key += (is[i]-noab-1)*offset; // TV
        offset *= (check)?noab:nvab;
      }
    }
    else {
      assert(0);
    }
    if (dist_type_ == dist_nwi)  {
      assert(Variables::intorb()!=0);
      assert(dim_ == 4);
      get_hash_block_i_(&d_a, buf, &size, &int_mb[d_a_offset], &key,
			&is[3], &is[2], &is[1], &is[0]); /* special case*/
    }
    else if(dist_type_ == dist_nwma) {
      double *dbl_mb = Variables::dbl_mb();
      get_hash_block_ma_(&dbl_mb[d_a], buf, &size, &int_mb[d_a_offset], &key);
    }
    else if(dist_type_ == dist_nw) {
      get_hash_block_(&d_a, buf, &size, &int_mb[d_a_offset], &key);
    }
    else {
      assert(0);
    }
#endif
  }

  // for t_assign only, somehow the key is different for non get_i case
  // void Tensor::get2(Integer d_a, double *buf, Integer size, Integer d_a_offset) {
  //   std::vector<Integer>& is = value_r_; // <-- not _value_r_
  //   Integer key = 0, offset = 1;
  //   Integer noab = Variables::noab();
  //   Integer nvab = Variables::nvab();
  //   Integer *int_mb = Variables::int_mb();
  //   int n = is.size(); // = dim_;

  //   assert(dim_type_ == dim_n);
  //   for (int i=n-1; i>=0; i--) {
  //     key += (is[i]-1) * offset;
  //     offset *= noab + nvab;
  //   }
  //   if(dist_type_ == dist_nwi) {
  //     get_hash_block_i_(&d_a, buf, &size, &int_mb[d_a_offset], &key, 
  //         &is[3], &is[2], &is[1], &is[0]);
  //   }
  //   else {
  //     get_hash_block_(&d_a, buf, &size, &int_mb[d_a_offset], &key);
  //   }
  // }

  void Tensor::get2(Integer d_a, std::vector<Integer> &pvalue_r, double *buf, Integer size, Integer d_a_offset) {
    std::vector<Integer>& is = pvalue_r; // <-- not _value_r_ or value_r
    Integer key = 0, offset = 1;
    Integer noab = Variables::noab();
    Integer nvab = Variables::nvab();
    Integer *int_mb = Variables::int_mb();
    int n = is.size(); // = dim_;

    assert(dim_type_ == dim_n);
    for (int i=n-1; i>=0; i--) {
      key += (is[i]-1) * offset;
      offset *= noab + nvab;
    }
    if(dist_type_ == dist_nwi) {
      get_hash_block_i_(&d_a, buf, &size, &int_mb[d_a_offset], &key, 
          &is[3], &is[2], &is[1], &is[0]);
    }
    else {
      get_hash_block_(&d_a, buf, &size, &int_mb[d_a_offset], &key);
    }
  }

  extern "C" {

    Tensor Tensor2(IndexName n1, IndexName n2, int e1, int e2, TensorType type,
		   DistType dt, DimType dm) {
      Index i1 = Index(n1,0);
      Index i2 = Index(n2,1); // prevent error!
      Index ids[] = {i1,i2};
      Tensor t = Tensor(2,ids,type,dt,dm);
      return t;
    }

    Tensor Tensor4(IndexName n1, IndexName n2, IndexName n3, IndexName n4,
		   int e1, int e2, int e3, int e4, TensorType type, DistType dt,
		   DimType dm) {
      assert(e1==0);
      Index i1 = Index(n1,e1);
      Index i2 = Index(n2,e2);
      Index i3 = Index(n3,e3);
      Index i4 = Index(n4,e4);
      Index ids[] = {i1,i2,i3,i4};
      Tensor t = Tensor(4,ids,type,dt,dm);
      return t;
    }

    Tensor Tensor6(IndexName n1, IndexName n2, IndexName n3, IndexName n4, IndexName n5, IndexName n6,
		   int e1, int e2, int e3, int e4, int e5, int e6, TensorType type,
		   DistType dt, DimType dm) {
      assert(e1==0);
      Index i1 = Index(n1,e1);
      Index i2 = Index(n2,e2);
      Index i3 = Index(n3,e3);
      Index i4 = Index(n4,e4);
      Index i5 = Index(n5,e5);
      Index i6 = Index(n6,e6);
      Index ids[] = {i1,i2,i3,i4,i5,i6};
      Tensor t = Tensor(6,ids,type,dt,dm);
      return t;
    }

  };

};
