#include "expression.h"
#include "t_mult.h"
#include "t_assign.h"

namespace ctce {

  void Assignment::init() {
    // const std::vector<IndexName>& name = tC_.name();
    // const std::vector<IndexName>& name = id2name(tC_.ids());
    const std::vector<IndexName>& name = cids_;
    // const std::vector<int>& gp = tC_.ext_sym_group();
    //const std::vector<int>& gp = ext_sym_group(tC_.ids());
    const std::vector<int>& gp = ext_sym_group(tC(), cids_);
    std::vector< std::vector<IndexName> > all;
    std::vector<IndexName> one;
    int prev=0, curr=0, n=name.size();

    for (int i=0; i<n; i++) {
      curr=gp[i];
      if (curr!=prev) {
        all.push_back(one);
        one.clear();
        prev=curr;
      }
      one.push_back(name[i]);
    }
    all.push_back(one);
    std::vector<triangular> vt(all.size());
    for (int i=0; i<all.size(); i++) {
      triangular tr(all[i]);
      vt[i]=tr;
    }
    out_itr_ = IterGroup<triangular>(vt,TRIG);
  }

  void Assignment::execute(int sync_ga) {
    assert(tA().attached() || tA().allocated());
    assert(tC().attached() || tC().allocated());
    Integer d_a, k_a_offset;
    Integer d_c, k_c_offset;
    d_a = tA().ga();
    k_a_offset = tA().offset_index();
    d_c = tC().ga();
    k_c_offset = tC().offset_index();

    t_assign3(&d_a, &k_a_offset, &d_c, &k_c_offset, *this, sync_ga);
  }

  void Multiplication::genMemPos() {
    // const std::vector<IndexName>& a = tA_.name();
    // const std::vector<IndexName>& b = tB_.name();
    const std::vector<IndexName>& a = id2name(a_ids);
    const std::vector<IndexName>& b = id2name(b_ids);
    std::vector<IndexName> a_ext, b_ext;
    std::vector<IndexName> a_mpos, b_mpos, c_mpos;
    int na = a.size();
    int nb = b.size();
    int af[na]; memset(af,0,na*sizeof(int));
    int bf[nb]; memset(bf,0,nb*sizeof(int));
    for (int i=0; i<na; i++)
      for (int j=0; j<nb; j++) {
        if (a[i]==b[j]) { // find common name
          af[i]=1; bf[j]=1;
        }
      }
    for (int i=0; i<na; i++) { 
      if (af[i]==0) a_ext.push_back(a[i]);
      else sum_ids_.push_back(a[i]);
    }
    for (int i=0; i<nb; i++) {
      if (bf[i]==0) b_ext.push_back(b[i]);
    }
    a_mpos = a_ext;
    a_mpos.insert(a_mpos.begin(),sum_ids_.begin(),sum_ids_.end());
    b_mpos = b_ext;
    b_mpos.insert(b_mpos.begin(),sum_ids_.begin(),sum_ids_.end());
    c_mpos = a_ext;
    c_mpos.insert(c_mpos.end(),b_ext.begin(),b_ext.end());
    reverse(a_mpos.begin(),a_mpos.end());
    reverse(b_mpos.begin(),b_mpos.end());
    reverse(c_mpos.begin(),c_mpos.end());
    // tC_.setMemPos(c_mpos);
    // tA_.setMemPos(a_mpos);
    // tB_.setMemPos(b_mpos);

    a_mem_pos = a_mpos;
    b_mem_pos = b_mpos;
    c_mem_pos = c_mpos;
  }

  void Multiplication::genTrigItr(IterGroup<triangular>& itr,
      const std::vector<int>& gp, const std::vector<IndexName>& name) {
    std::vector< std::vector<IndexName> > all;
    std::vector<IndexName> one;
    int prev=0, curr=0, n=name.size();
    for (int i=0; i<n; i++) {
      curr=gp[i];
      if (curr!=prev) {
        all.push_back(one);
        one.clear();
        prev=curr;
      }
      one.push_back(name[i]);
    }
    all.push_back(one);
    std::vector<triangular> vt(all.size());
    for (int i=0; i<all.size(); i++) {
      triangular tr(all[i]);
      vt[i]=tr;
    }
    itr = IterGroup<triangular>(vt,TRIG);
  }

  void Multiplication::genSumGroup() {
    //assert(sum_ids_.size()>0);  ccsd(t) singles has no summation indices
    if (sum_ids_.size()==0) {
      sum_itr_ = IterGroup<triangular>(); // empty iterator
      return;
    }
    std::vector<int> s_gp(sum_ids_.size());
    for (int i=0; i<sum_ids_.size(); i++) {
      s_gp[i] = getIndexType(sum_ids_[i]);
    }
    if (s_gp[0]==hIndex) { // should start with pIndex, if not then flip all numbers
      for (int i=0; i<s_gp.size(); i++) s_gp[i]=s_gp[i]^1; // flip 0 and 1 if first is 1
    }
    //    std::cout << "s_gp:" << s_gp << std::endl;
    setSumItr(s_gp);
  }

  void Multiplication::genOutGroup() {

    // const std::vector<IndexName>& c = tC_.name();
    // const std::vector<IndexName>& a = tA_.name();
    // const std::vector<IndexName>& b = tB_.name();
    const std::vector<IndexName>& c = id2name(c_ids);
    const std::vector<IndexName>& a = id2name(a_ids);
    const std::vector<IndexName>& b = id2name(b_ids);
    int n = tC().dim();
    assert(n>0);
    int from[n];
    for (int i=0; i<n; i++) {
      for (int j=0; j<tA().dim(); j++)
        if (c[i]==a[j]) from[i]=0;
      for (int j=0; j<tB().dim(); j++)
        if (c[i]==b[j]) from[i]=1;
    }
    // std::vector<int> c_ext = tC_.ext_sym_group();
    std::vector<int> c_ext = ext_sym_group(c_ids);
    std::vector<int> group(n);
    int offset=0;
    int dirty[n]; memset(dirty,0,n*sizeof(int));
    for (int i=0; i<c_ext.size(); i++) {
      if (dirty[i]==0) {
        group[i]=offset;
        dirty[i]=1;
        offset++;
      }
      for (int j=0; j<c_ext.size(); j++) { // find common
        if (dirty[j]) continue;
        if (c_ext[i]==c_ext[j] && from[i]==from[j]) {
          group[j]=group[i];
          dirty[j]=1;
        }
      }
    }
    //    std::cout << "out_group:" << group << std::endl;
    setOutItr(group);

    std::vector<int> cp_group;
    int curr_e = c_ext[0], prev_e = c_ext[0]; // {0,0,1,1}
    int curr_o = group[0], prev_o = group[0]; // {0,0,1,2}  --> 4,4,3
    int curr_size = 1;
    for (int i=1; i<group.size(); i++) {
      curr_e = c_ext[i];
      curr_o = group[i];
      if (curr_e != prev_e) {
        cp_group.push_back(curr_size);
        curr_size=1;
      }
      else if (curr_o == prev_o) {
        cp_group.push_back(curr_size);
        curr_size=1;
      }
      else if (curr_o != prev_o) {
        curr_size+=1;
      }
      prev_e = curr_e;
      prev_o = curr_o;
    }
    cp_group.push_back(curr_size);

    //    std::cout << "cp_group: ";
    //    std::cout << cp_group << std::endl;

    // should automatically generate cp_itr instead of using dummy ones
    int cpn = cp_group.size();
    std::vector<CopyIter> vd(cpn);
    for (int i=0; i<cpn; i++) {
      if (cp_group[i]==1) vd[i] = Dummy::type4();
      if (cp_group[i]==2) vd[i] = Dummy::type3();
    }
    cp_itr_ = IterGroup<CopyIter>(vd, COPY);
  }

  void Multiplication::setSumItr(const std::vector<int>& gp) {
    genTrigItr(sum_itr_,gp,sum_ids_);
  }

  void Multiplication::setOutItr(const std::vector<int>& gp) {
    // genTrigItr(out_itr_,gp,tC_.name());
    genTrigItr(out_itr_,gp,id2name(c_ids));
  }

  void Multiplication::setCopyItr(const std::vector<int>& gp) {
    int n = gp.size();
    std::vector<CopyIter> vd(n);
    for (int i=0; i<n; i++) {
      if (gp[i]==1) vd[i] = Dummy::type1();
      if (gp[i]==2) vd[i] = Dummy::type2();
      if (gp[i]==3) vd[i] = Dummy::type3();
      if (gp[i]==4) vd[i] = Dummy::type4();
    }
    cp_itr_ = IterGroup<CopyIter>(vd, COPY);
  }

  void Multiplication::execute(int sync_ga) {
    assert(tA().attached() || tA().allocated());
    assert(tB().attached() || tB().allocated());
    assert(tC().attached() || tC().allocated());
    Integer d_a, k_a_offset;
    Integer d_b, k_b_offset;
    Integer d_c, k_c_offset;
    d_a = tA().ga();
    k_a_offset = tA().offset_index();
    d_b = tB().ga();
    k_b_offset = tB().offset_index();
    d_c = tC().ga();
    k_c_offset = tC().offset_index();

    t_mult4(&d_a, &k_a_offset, &d_b, &k_b_offset, &d_c, &k_c_offset, *this, sync_ga);
  }
};
