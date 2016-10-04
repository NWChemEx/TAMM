#include <vector>
#include <map>
#include <set>
#include "tensor.h"
#include "expression.h"
#include "input.h"
// #include "equations.h"
#include <iostream>
#include <sstream>
#include <string>

namespace tamm {
  struct Equations {
    std::vector<RangeEntry> range_entries;
    std::vector<IndexEntry> index_entries;
    std::map<std::string, TensorEntry> tensor_entries;
    std::vector<OpEntry> op_entries;
  };
}

namespace tamm {

  std::string index_to_string(const Equations &eqs, 
                              int nids, const int *ids) {
    std::stringstream ss;
    ss<< "[";
    for(int i=0; i<nids-1; i++) {
      if(ids[i]<IndexNum) {
        ss<< eqs.index_entries[ids[i]].name << std::string(",");
      }
      else {
        ss<< "i" <<ids[i]<<",";
      }
    }
    if(nids>0) {
      if(ids[nids-1]<IndexNum) {
        ss<< eqs.index_entries[ids[nids-1]].name;
      }
      else {
        ss<< "i" <<ids[nids-1];
      }
      //ss<<eqs.index_entries[ids[nids-1]].name;
    }
    ss<< "]";
    return ss.str();
    // ret += "[";
    // for(int i=0; i<nids-1; i++) {
    //   ret += eqs.index_entries[ids[i]].name+std::string(",");
    // }
    // if(nids>0) {
    //   ret += eqs.index_entries[ids[nids-1]].name;
    // }
    // ret += "]";
    // return ret;
  }

  std::string index_to_string(const Equations &eqs, 
                              const std::vector<int> &ids) {
    int nids = ids.size();
    std::stringstream ss;
    std::string ret;
    ss<< "[";
#if 1
    for(int i=0; i<nids-1; i++) {
      if(ids[i]<IndexNum) {
        ss<< eqs.index_entries[ids[i]].name << std::string(",");
      }
      else {
        ss<< "i" <<ids[i]<<",";
      }
    }
    if(nids>0) {
      if(ids[nids-1]<IndexNum) {
        ss<< eqs.index_entries[ids[nids-1]].name;
      }
      else {
        ss<< "i" <<ids[nids-1];
      }
      //ss<<eqs.index_entries[ids[nids-1]].name;
    }
#endif
    ss<< "]";
    return ss.str();
  }

  void pretty_print(Equations &eqs) {
    ostream &os = std::cout;
    //for(int i=0; i<eqs.tensor_entries.size(); i++) {
    for(std::map<std::string, TensorEntry>::iterator i = eqs.tensor_entries.begin(); i != eqs.tensor_entries.end(); i++){
      const TensorEntry& te = eqs.tensor_entries[i->first];
      os<<"array "<<te.name<<"[";
      for(int j=0; j<te.nupper-1; j++) {
        os<<eqs.range_entries[te.range_ids[j]].name<<",";
      }
      if(te.nupper>0) {
        os<<eqs.range_entries[te.range_ids[te.nupper-1]].name;
      }
      os<<"][";
      for(int j=te.nupper; j<te.ndim-1; j++) {
        os<<eqs.range_entries[te.range_ids[j]].name<<",";        
      }
      if(te.ndim>te.nupper) {
        os<<eqs.range_entries[te.range_ids[te.ndim-1]].name;
      }
      os<<"];\n";
    }
    os<<"\n";
    for(int i=0; i<eqs.op_entries.size(); i++) {
      const OpEntry &ope = eqs.op_entries[i];
      std::map<std::string, TensorEntry> &tes = eqs.tensor_entries;
      //int ta, tb, tc;
      switch(ope.optype) {
      case OpTypeAdd:
        os<<"op" << ope.op_id << ": " << tes[ope.add.tc].name
          <<index_to_string(eqs, tes[ope.add.tc].ndim, ope.add.tc_ids)
          <<" += "<<ope.add.alpha<<" * "
          <<tes[ope.add.ta].name
          <<index_to_string(eqs, tes[ope.add.ta].ndim, ope.add.ta_ids)
          <<";"<<endl;
          break;
      case OpTypeMult:
        os<<"op" << ope.op_id << ": " <<tes[ope.mult.tc].name
          <<index_to_string(eqs, tes[ope.mult.tc].ndim, ope.mult.tc_ids)
          <<" += "<<ope.mult.alpha<<" * "
          <<tes[ope.mult.ta].name
          <<index_to_string(eqs, tes[ope.mult.ta].ndim, ope.mult.ta_ids)
          <<" * "<<tes[ope.mult.tb].name
          <<index_to_string(eqs, tes[ope.mult.tb].ndim, ope.mult.tb_ids)
          <<";"<<endl;
          break;
      default:
        assert(0);
      }
    }
  }

  void compute_deps(const Equations &eqs, vector<vector<int> > &deps) {
    deps.clear();
    deps.resize(eqs.op_entries.size());
    for(int i=0; i<deps.size(); i++) {
      for(int j=0; j<i; j++) {
        //int ra1=0, ra2=0, wa=0, rb1=0, rb2=0, wb=0;
        std::string ra1, ra2, wa, rb1, rb2, wb;
        const vector<OpEntry> &ops = eqs.op_entries;
        switch(ops[i].optype) {
        case OpTypeAdd:
          wa = ops[i].add.tc;
          ra1 = ops[i].add.ta;
          break;
        case OpTypeMult:
          wa = ops[i].mult.tc;
          ra1 = ops[i].mult.ta;
          ra2 = ops[i].mult.tb;
          break;
        default:
          assert(0);
        }
        switch(ops[j].optype) {
        case OpTypeAdd:
          wb = ops[j].add.tc;
          rb1 = ops[j].add.ta;
          break;
        case OpTypeMult:
          wb = ops[j].mult.tc;
          rb1 = ops[j].mult.ta;
          rb2 = ops[j].mult.tb;
          break;
        default:
          assert(0);
        }
        if(!ra1.compare(wb) || !ra2.compare(wb) ||
           !rb1.compare(wa) || !rb2.compare(wa)) {
          deps[i].push_back(j);
        }
      }
    }
  }

#if 0
  void print_ilp(const Equations &eqs) {
    //@BUG @FIXME compute actual tensor sizes and memory bound
    vector<size_t> tensor_sizes(eqs.tensor_entries.size(), 0);
    size_t membound = 100;
    vector<vector<int> > deps;
    compute_deps(eqs, deps);
    ostream &os = std::cout;    
    os<<"#### PARAMETERS ######"<<endl
      <<"#Number of tensors"<<endl
      <<"param NT > 0 integer;"<<endl
      <<"#Number of operations"<<endl
      <<"param NO > 0 integer;"<<endl
      <<"#Maximum number of levels"<<endl
      <<"param MAXLEVELS > 0 integer;"<<endl
      <<"#Memory bound"<<endl
      <<"param MEMBOUND > 0 integer;"<<endl
      <<"#Set of tensors"<<endl
      <<"set TENSORS := 1..NT;"<<endl
      <<"#Set of operations"<<endl
      <<"set OPS := 1..NO;"<<endl
      <<"#size of each tensor"<<endl
      <<"param size {TENSORS};"<<endl
      <<endl
      <<"##### VARIABLES ########"<<endl
      <<"#Level assigned to each operation"<<endl
      <<"var Level {OPS} >= 1;"<<endl
      <<
      ;


    os<<"data;"<<endl
      <<"param NT := "<<eqs.tensor_entries.size()<<endl
      <<"param NO := "<<eqs.op_entries.size()<<endl
      <<"param MAXLEVELS := "<<eqs.op_entries.size()<<endl
      <<"param MEMBOUND := "<<membound<<endl
      <<"param: size := "<<endl
      ;
    for(int i=0; i<eqs.tensor_entries.size(); i++) {
      os<<(i+1)<<"\t"<<tensor_sizes[i]<<endl;
    }
    
    
  }
#endif
#if 0
  void io_tensors(const std::vector<TensorEntry> &tes,
                      const std::vector<OpEntry> &opes,
                      std::vector<int> &outputs) {
    std::vector<bool> is_output(tes.size(), true);
    std::vector<bool> is_input(tes.size(), true);
    std::vector<bool> is_input(tes.size(), true);
    int ta, tb, tc;
    for(int i=0; i<opes[i].size(); i++) {
      switch(opes[i].optype) {
      case OpTypeAdd:
        
        break;
      case OpTypeMult:
        break;
      default:
        printf("Unsupported operation\n");
        assert(0);
      }
    }
  }

  void input_tensors(const std::vector<Tensor> &tensors,
                     const std::vector<Operation> &ops,
                     std::vector<int> &inputs) {
  }
#endif
  struct TensorUse {
    int tensor_id;
    vector<int> ids; //indices into Equation::index_entries
  };

  struct Term {
    double alpha;
    vector<TensorUse> trefs;
    Term() : alpha(1) {}
  };  

  Term cross(const Term& t1, const Term& t2) {
    Term t;
    t.alpha = t1.alpha * t2.alpha;
    t.trefs = t1.trefs;
    t.trefs.insert(t.trefs.end(), t2.trefs.begin(), t2.trefs.end());
    return t;
  }

  void canonicalize(Equations &eqs) {
    int tmp_num = IndexNum + 1;
    return;
#if 1
    for(int i=0; i<eqs.op_entries.size(); i++) {
      OpEntry &ope = eqs.op_entries[i];
      if(ope.optype == OpTypeAdd) continue;
      assert(ope.optype == OpTypeMult);
      std::vector<int> rename(IndexNum, -1);
      TensorEntry &cte = eqs.tensor_entries[ope.mult.tc];
      TensorEntry &ate = eqs.tensor_entries[ope.mult.ta];
      TensorEntry &bte = eqs.tensor_entries[ope.mult.tb];
      for(int j=0; j<cte.ndim; j++) {
        //external indices
        rename[ope.mult.tc_ids[j]] = ope.mult.tc_ids[j];
      }      
      for(int j=0; j<ate.ndim; j++) {
        if(rename[ope.mult.ta_ids[j]] == -1) {
          rename[ope.mult.ta_ids[j]] = tmp_num++;
        }
        ope.mult.ta_ids[j] = rename[ope.mult.ta_ids[j]];
      }
      for(int j=0; j<bte.ndim; j++) {
        //every index is in either tc or tb
        assert(rename[ope.mult.tb_ids[j]] != -1);
        ope.mult.ta_ids[j] = rename[ope.mult.ta_ids[j]];
      }
    }
#endif
  }

//  void print_flatten(Equations &eqs, int opid) {
//    using std::vector;
//    assert(opid >=0 && opid<eqs.op_entries.size());
//
//    //canonicalize(eqs);
//    vector<vector<Term> > op_defs(opid+1);
//    vector<int> tensor_defs(eqs.tensor_entries.size(),-1); //opid defining the tensor
//    std::map<std::string, TensorEntry> &tes = eqs.tensor_entries;
//    const vector<OpEntry> &opes = eqs.op_entries;
//
//    for(int i=0; i<opid+1; i++) {
//      const OpEntry &ope = eqs.op_entries[i];
//      std::string ta, tb, tc;
//      double alpha;
//      vector<Term> a_terms, b_terms;
//      switch(ope.optype) {
//      case OpTypeAdd:
//        ta = ope.add.ta;
//        tc = ope.add.tc;
//        alpha = ope.add.alpha;
//        if(tensor_defs[tc]!=-1) {
//          op_defs[i].insert(op_defs[i].end(),
//                            op_defs[tensor_defs[tc]].begin(),
//                            op_defs[tensor_defs[tc]].end());
//        }
//        if(tensor_defs[ta]!=-1) {
//          vector<Term> terms = op_defs[tensor_defs[ta]];
//          for(int j=0; j<terms.size(); j++) {
//            terms[i].alpha *= alpha;
//          }
//          op_defs[i].insert(op_defs[i].end(), terms.begin(), terms.end());
//        }
//        else {
//          TensorUse tu;
//          tu.tensor_id = ta;
//          int nadim = eqs.tensor_entries[ta].ndim;
//          tu.ids.insert(tu.ids.end(),ope.add.ta_ids, ope.add.ta_ids+nadim);
//          Term term;
//          term.alpha = alpha;
//          term.trefs.push_back(tu);
//          op_defs[i].push_back(term);
//        }
//        tensor_defs[tc] = i;
//        break;
//      case OpTypeMult:
//        ta = ope.mult.ta;
//        tb = ope.mult.tb;
//        tc = ope.mult.tc;
//        alpha = ope.mult.alpha;
//        if(tensor_defs[tc]!=-1) {
//          op_defs[i].insert(op_defs[i].end(),
//                            op_defs[tensor_defs[tc]].begin(),
//                            op_defs[tensor_defs[tc]].end());
//        }
//        if(tensor_defs[ta]!=-1) {
//          a_terms = op_defs[tensor_defs[ta]];
//        }
//        else {
//#if 1
//          TensorUse tu;
//          tu.tensor_id = ta;
//          int nadim = eqs.tensor_entries[ta].ndim;
//          tu.ids.insert(tu.ids.end(),ope.mult.ta_ids, ope.mult.ta_ids+nadim);
//          Term term;
//          term.alpha = 1;
//          term.trefs.push_back(tu);
//          a_terms.push_back(term);
//#endif
//        }
//        if(tensor_defs[tb]!=-1) {
//          b_terms = op_defs[tensor_defs[tb]];
//        }
//        else {
//#if 1
//          TensorUse tu;
//          tu.tensor_id = tb;
//          int nbdim = eqs.tensor_entries[tb].ndim;
//          tu.ids.insert(tu.ids.end(),ope.mult.tb_ids, ope.mult.tb_ids+nbdim);
//          Term term;
//          term.alpha = 1;
//          term.trefs.push_back(tu);
//          b_terms.push_back(term);
//#endif
//        }
//        for(int a=0; a<a_terms.size(); a++) {
//          for(int b=0; b<b_terms.size(); b++) {
//            Term t = cross(a_terms[a], b_terms[b]);
//            t.alpha *= alpha;
//            op_defs[i].push_back(t);
//          }
//        }
//        tensor_defs[tc] = i;
//        break;
//      default:
//        assert(0);
//      }
//    }
//
//    cout<<"opid="<<opid<<endl;
//    const OpEntry& ope = opes[opid];
//    std::string tc = (ope.optype==OpTypeAdd)? ope.add.tc : ope.mult.tc;
//    const int *tc_ids = (ope.optype==OpTypeAdd)? ope.add.tc_ids : ope.mult.tc_ids;
//    cout<<tes[tc].name
//        <<index_to_string(eqs, tes[tc].ndim, tc_ids)
//        <<" += ";
//    for(int i=0; i<op_defs[opid].size(); i++)
//      {
//      Term &term = op_defs[opid][i];
//      cout<<endl<<" +"<<term.alpha;
//      for(int j=0; j<term.trefs.size(); j++) {
//        TensorUse &tu = term.trefs[j];
//        cout<<" * "<<tes[tu.tensor_id].name
//            <<index_to_string(eqs, tu.ids);
//      }
//    }
//    cout<<";"<<endl;
//  }


} /*tamm*/
