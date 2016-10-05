#include "preprocess.h"
#include "expression.h"

namespace tamm {

void genTrigIter(IterGroup<triangular>& trig_itr,
                 const std::vector<IndexName>& name,
                 const std::vector<int>& group) {
  std::vector<std::vector<IndexName> > all;
  std::vector<IndexName> one;
  int prev = 0, curr = 0, n = name.size();
  for (int i = 0; i < n; i++) {
    curr = group[i];
    if (curr != prev) {
      all.push_back(one);
      one.clear();
      prev = curr;
    }
    one.push_back(name[i]);
  }
  all.push_back(one);
  std::vector<triangular> vt(all.size());
  for (int i = 0; i < all.size(); i++) {
    triangular tr(all[i]);
    vt[i] = tr;
  }
  trig_itr = IterGroup<triangular>(vt, TRIG);
}

void genAntiIter(const std::vector<size_t>& vtab, IterGroup<antisymm>& ext_itr,
                 const Tensor& tC, const Tensor& tA, const Tensor& tB) {
  const std::vector<IndexName>& c = id2name(tC.ids());
  const std::vector<IndexName>& a = id2name(tA.ids());
  const std::vector<IndexName>& b = id2name(tB.ids());
  const std::vector<Index>& c_ids = tC.ids();
  std::vector<int> from(tC.dim());
  for (int i = 0; i < tC.dim(); i++) {
    for (int j = 0; j < tA.dim(); j++)
      if (c[i] == a[j]) from[i] = 0;
    for (int j = 0; j < tB.dim(); j++)
      if (c[i] == b[j]) from[i] = 1;
  }
  std::vector<std::vector<IndexName> > all;
  std::vector<IndexName> one;
  std::vector<std::vector<int> > all_f;
  std::vector<int> one_f;
  int prev = 0, curr = 0;

  const vector<int> esg = ext_sym_group(tC, c);
  for (int i = 0; i < tC.dim(); i++) {
    // curr = c_ids[i].ext_sym_group();
    curr = esg[i];
    if (curr != prev) {
      all.push_back(one);
      all_f.push_back(one_f);
      one.clear();
      one_f.clear();
      prev = curr;
    }
    one.push_back(c[i]);
    one_f.push_back(from[i]);
  }
  all.push_back(one);
  all_f.push_back(one_f);
  std::vector<antisymm> va(all.size());
  for (int i = 0; i < all.size(); i++) {
    int fb = 0;
    for (int j = 0; j < all_f[i].size(); j++) fb += all_f[i][j];
    int fa = all_f[i].size() - fb;
    va[i] = antisymm(vtab, all[i], fa, fb);
  }
  ext_itr = IterGroup<antisymm>(va, ANTI);
}

} /* namespace tamm */
