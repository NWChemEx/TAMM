#include "ccse_tensors.hpp"
#include <tamm/tamm_git.hpp>

using CCEType = double;
CCSE_Tensors<CCEType> _a021;
Tensor<CCEType>       a22_abab;
TiledIndexSpace       o_alpha, v_alpha, o_beta, v_beta;

Tensor<CCEType>       _a01V, _a02V, _a007V;
CCSE_Tensors<CCEType> _a01, _a02, _a03, _a04, _a05, _a06, _a001, _a004, _a006, _a008, _a009, _a017,
  _a019, _a020; //_a022

Tensor<CCEType> i0_temp, t2_aaaa_temp; // CS only

template<typename T>
std::tuple<std::vector<T>, Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>>
setupTensors_cs(ExecutionContext& ec, TiledIndexSpace& MO, Tensor<T> d_f1) {
  // auto rank = ec.pg().rank();

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();

  TiledIndexSpace o_alpha, v_alpha, o_beta, v_beta;
  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(oatiles, otiles)};
  v_beta  = {MO("virt"), range(vatiles, vtiles)};

  std::vector<T> p_evl_sorted = tamm::diagonal(d_f1);

  Tensor<T> d_r1{{v_alpha, o_alpha}, {1, 1}};
  Tensor<T> d_r2{{v_alpha, v_beta, o_alpha, o_beta}, {2, 2}};

  Tensor<T>::allocate(&ec, d_r1, d_r2);

  Tensor<T> d_t1{{v_alpha, o_alpha}, {1, 1}};
  Tensor<T> d_t2{{v_alpha, v_beta, o_alpha, o_beta}, {2, 2}};

  Tensor<T>::allocate(&ec, d_t1, d_t2);

  return std::make_tuple(p_evl_sorted, d_t1, d_t2, d_r1, d_r2);
}

template<typename T>
void ccsd_e_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI, Tensor<T>& de,
               const Tensor<T>& t1_aa, const Tensor<T>& t2_abab, const Tensor<T>& t2_aaaa,
               std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se) {
  auto [cind] = CI.labels<1>("all");

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb]        = v_beta.labels<1>("all");
  auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  auto [h1_ob]        = o_beta.labels<1>("all");

  // f1_se     = {f1_oo,f1_ov,f1_vv}
  // chol3d_se = {chol3d_oo,chol3d_ov,chol3d_vv}
  auto f1_ov     = f1_se[1];
  auto chol3d_ov = chol3d_se[1];

  // clang-format off
  sch
    (t2_aaaa_temp()=0)
    .exact_copy(t2_aaaa(p1_va, p2_va, h1_oa, h2_oa), t2_abab(p1_va, p2_va, h1_oa, h2_oa))
    (t2_aaaa_temp() = t2_aaaa(),
    "t2_aaaa_temp() = t2_aaaa()")
    (t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) += -1.0 * t2_aaaa_temp(p2_va,p1_va,h1_oa,h2_oa),
    "t2_aaaa(p1_va,p2_va,h1_oa,h2_oa) += -1.0 * t2_aaaa_temp(p2_va,p1_va,h1_oa,h2_oa)")
    (t2_aaaa_temp(p1_va,p2_va,h1_oa,h2_oa) +=  1.0 * t2_aaaa(p2_va,p1_va,h2_oa,h1_oa),
    "t2_aaaa_temp(p1_va,p2_va,h1_oa,h2_oa) +=  1.0 * t2_aaaa(p2_va,p1_va,h2_oa,h1_oa)")

    (_a01V(cind) = t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "_a01V(cind) = t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (_a02("aa")(h1_oa, h2_oa, cind)    = t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h2_oa, p1_va, cind),
    "_a02( aa )(h1_oa, h2_oa, cind)    = t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h2_oa, p1_va, cind)")
    (_a03("aa")(h2_oa, p2_va, cind) = t2_aaaa_temp(p2_va, p1_va, h2_oa, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "_a03( aa )(h2_oa, p2_va, cind) = t2_aaaa_temp(p2_va, p1_va, h2_oa, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (de()  =  2.0 * _a01V() * _a01V(),
    "de()  =  2.0 * _a01V() * _a01V()")
    (de() += -1.0 * _a02("aa")(h1_oa, h2_oa, cind) * _a02("aa")(h2_oa, h1_oa, cind),
    "de() += -1.0 * _a02( aa )(h1_oa, h2_oa, cind) * _a02( aa )(h2_oa, h1_oa, cind)")
    (de() +=  1.0 * _a03("aa")(h1_oa, p1_va, cind) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "de() +=  1.0 * _a03( aa )(h1_oa, p1_va, cind) * chol3d_ov( aa )(h1_oa, p1_va, cind)")
    (de() +=  2.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h1_oa, p1_va),
    "de() +=  2.0 * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h1_oa, p1_va)") // NEW TERM
    ;
  // clang-format on
}

template<typename T>
void ccsd_t1_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                Tensor<T>& i0_aa, const Tensor<T>& t1_aa, const Tensor<T>& t2_abab,
                std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se) {
  auto [cind] = CI.labels<1>("all");
  auto [p2]   = MO.labels<1>("virt");
  auto [h1]   = MO.labels<1>("occ");

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb]        = v_beta.labels<1>("all");
  auto [h1_oa, h2_oa] = o_alpha.labels<2>("all");
  auto [h1_ob]        = o_beta.labels<1>("all");

  // f1_se     = {f1_oo,f1_ov,f1_vv}
  // chol3d_se = {chol3d_oo,chol3d_ov,chol3d_vv}
  auto f1_oo     = f1_se[0];
  auto f1_ov     = f1_se[1];
  auto f1_vv     = f1_se[2];
  auto chol3d_oo = chol3d_se[0];
  auto chol3d_ov = chol3d_se[1];
  auto chol3d_vv = chol3d_se[2];

  // clang-format off
  sch
    (i0_aa(p2_va, h1_oa)             =  1.0 * f1_ov("aa")(h1_oa, p2_va),
    "i0_aa(p2_va, h1_oa)             =  1.0 * f1_ov( aa )(h1_oa, p2_va)")
    (_a01("aa")(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h2_oa, p1_va, cind),
    "_a01( aa )(h2_oa, h1_oa, cind)  =  1.0 * t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h2_oa, p1_va, cind)")                 // ovm
    (_a02V(cind)                     =  2.0 * t1_aa(p1_va, h1_oa) * chol3d_ov("aa")(h1_oa, p1_va, cind),
    "_a02V(cind)                     =  2.0 * t1_aa(p1_va, h1_oa) * chol3d_ov( aa )(h1_oa, p1_va, cind)")                 // ovm
    // (_a02V(cind)                  =  2.0 * _a01("aa")(h1_oa, h1_oa, cind))
    (_a05("aa")(h2_oa, p1_va)        = -1.0 * chol3d_ov("aa")(h1_oa, p1_va, cind) * _a01("aa")(h2_oa, h1_oa, cind),
    "_a05( aa )(h2_oa, p1_va)        = -1.0 * chol3d_ov( aa )(h1_oa, p1_va, cind) * _a01( aa )(h2_oa, h1_oa, cind)")      // o2vm
    (_a05("aa")(h2_oa, p1_va)       +=  1.0 * f1_ov("aa")(h2_oa, p1_va),
    "_a05( aa )(h2_oa, p1_va)       +=  1.0 * f1_ov( aa )(h2_oa, p1_va)") // NEW TERM
    // .exact_copy(_a05_bb(h1_ob,p1_vb),_a05_aa(h1_ob,p1_vb))

    (_a06("aa")(p1_va, h1_oa, cind)  = -1.0 * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa) * chol3d_ov("aa")(h2_oa, p2_va, cind),
    "_a06( aa )(p1_va, h1_oa, cind)  = -1.0 * t2_aaaa_temp(p1_va, p2_va, h1_oa, h2_oa) * chol3d_ov( aa )(h2_oa, p2_va, cind)") // o2v2m
    (_a04("aa")(h2_oa, h1_oa)        = -1.0 * f1_oo("aa")(h2_oa, h1_oa),
    "_a04( aa )(h2_oa, h1_oa)        = -1.0 * f1_oo( aa )(h2_oa, h1_oa)") // MOVED TERM
    (_a04("aa")(h2_oa, h1_oa)       +=  1.0 * chol3d_ov("aa")(h2_oa, p1_va, cind) * _a06("aa")(p1_va, h1_oa, cind),
    "_a04( aa )(h2_oa, h1_oa)       +=  1.0 * chol3d_ov( aa )(h2_oa, p1_va, cind) * _a06( aa )(p1_va, h1_oa, cind)")   // o2vm
    (_a04("aa")(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va),
    "_a04( aa )(h2_oa, h1_oa)       += -1.0 * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h2_oa, p1_va)") // NEW TERM
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04("aa")(h2_oa, h1_oa),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p2_va, h2_oa) * _a04( aa )(h2_oa, h1_oa)")                         // o2v
    (i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_ov("aa")(h2_oa, p1_va, cind) * _a02V(cind),
    "i0_aa(p1_va, h2_oa)            +=  1.0 * chol3d_ov( aa )(h2_oa, p1_va, cind) * _a02V(cind)")                      // ovm
    (i0_aa(p1_va, h2_oa)            +=  1.0 * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * _a05("aa")(h1_oa, p2_va),
    "i0_aa(p1_va, h2_oa)            +=  1.0 * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * _a05( aa )(h1_oa, p2_va)")
    (i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_vv("aa")(p2_va, p1_va, cind) * _a06("aa")(p1_va, h1_oa, cind),
    "i0_aa(p2_va, h1_oa)            += -1.0 * chol3d_vv( aa )(p2_va, p1_va, cind) * _a06( aa )(p1_va, h1_oa, cind)")   // ov2m
    (_a06("aa")(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv("aa")(p2_va, p1_va, cind),
    "_a06( aa )(p2_va, h2_oa, cind) += -1.0 * t1_aa(p1_va, h2_oa) * chol3d_vv( aa )(p2_va, p1_va, cind)")              // ov2m
    (i0_aa(p1_va, h2_oa)            += -1.0 * _a06("aa")(p1_va, h2_oa, cind) * _a02V(cind),
    "i0_aa(p1_va, h2_oa)            += -1.0 * _a06( aa )(p1_va, h2_oa, cind) * _a02V(cind)")                           // ovm
    (_a06("aa")(p2_va, h1_oa, cind) += -1.0 * t1_aa(p2_va, h1_oa) * _a02V(cind),
    "_a06( aa )(p2_va, h1_oa, cind) += -1.0 * t1_aa(p2_va, h1_oa) * _a02V(cind)")                                      // ovm
    (_a06("aa")(p2_va, h1_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01("aa")(h2_oa, h1_oa, cind),
    "_a06( aa )(p2_va, h1_oa, cind) +=  1.0 * t1_aa(p2_va, h2_oa) * _a01( aa )(h2_oa, h1_oa, cind)")                   // o2vm
    (_a01("aa")(h2_oa, h1_oa, cind) +=  1.0 * chol3d_oo("aa")(h2_oa, h1_oa, cind),
    "_a01( aa )(h2_oa, h1_oa, cind) +=  1.0 * chol3d_oo( aa )(h2_oa, h1_oa, cind)")                                    // o2m
    (i0_aa(p2_va, h1_oa)            +=  1.0 * _a01("aa")(h2_oa, h1_oa, cind) * _a06("aa")(p2_va, h2_oa, cind),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * _a01( aa )(h2_oa, h1_oa, cind) * _a06( aa )(p2_va, h2_oa, cind)")        // o2vm
    // (i0_aa(p2_va, h1_oa)            += -1.0 * t1_aa(p2_va, h2_oa) * f1_oo("aa")(h2_oa, h1_oa), // MOVED ABOVE
    // "i0_aa(p2_va, h1_oa)            += -1.0 * t1_aa(p2_va, h2_oa) * f1_oo( aa )(h2_oa, h1_oa)")                        // o2v
    (i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p1_va, h1_oa) * f1_vv("aa")(p2_va, p1_va),
    "i0_aa(p2_va, h1_oa)            +=  1.0 * t1_aa(p1_va, h1_oa) * f1_vv( aa )(p2_va, p1_va)")                        // ov2
    ;
  // clang-format on
}

template<typename T>
void ccsd_t2_cs(Scheduler& sch, const TiledIndexSpace& MO, const TiledIndexSpace& CI,
                Tensor<T>& i0_abab, const Tensor<T>& t1_aa, Tensor<T>& t2_abab, Tensor<T>& t2_aaaa,
                std::vector<CCSE_Tensors<T>>& f1_se, std::vector<CCSE_Tensors<T>>& chol3d_se) {
  auto [cind]   = CI.labels<1>("all");
  auto [p3, p4] = MO.labels<2>("virt");
  auto [h1, h2] = MO.labels<2>("occ");

  auto [p1_va, p2_va, p3_va] = v_alpha.labels<3>("all");
  auto [p1_vb, p2_vb]        = v_beta.labels<2>("all");
  auto [h1_oa, h2_oa, h3_oa] = o_alpha.labels<3>("all");
  auto [h1_ob, h2_ob]        = o_beta.labels<2>("all");

  // f1_se     = {f1_oo,f1_ov,f1_vv}
  // chol3d_se = {chol3d_oo,chol3d_ov,chol3d_vv}
  auto f1_oo     = f1_se[0];
  auto f1_ov     = f1_se[1];
  auto f1_vv     = f1_se[2];
  auto chol3d_oo = chol3d_se[0];
  auto chol3d_ov = chol3d_se[1];
  auto chol3d_vv = chol3d_se[2];
  auto hw        = sch.ec().exhw();
  // auto rank      = sch.ec().pg().rank();

  //_a022("abab")(p1_va,p2_vb,p2_va,p1_vb) = _a021("aa")(p1_va,p2_va,cind) *
  //_a021("bb")(p2_vb,p1_vb,cind)
  Tensor<T>        a22_abab_tmp{v_alpha, v_beta, v_alpha, v_beta};
  LabeledTensor<T> lhs_  = a22_abab_tmp(p1_va, p2_vb, p2_va, p1_vb);
  LabeledTensor<T> rhs1_ = _a021("aa")(p1_va, p2_va, cind);
  LabeledTensor<T> rhs2_ = _a021("bb")(p2_vb, p1_vb, cind);

  // mult op constructor
  auto lhs_lbls  = lhs_.labels();
  auto rhs1_lbls = rhs1_.labels();
  auto rhs2_lbls = rhs2_.labels();

  IntLabelVec lhs_int_labels_;
  IntLabelVec rhs1_int_labels_;
  IntLabelVec rhs2_int_labels_;

  auto labels{lhs_lbls};
  labels.insert(labels.end(), rhs1_lbls.begin(), rhs1_lbls.end());
  labels.insert(labels.end(), rhs2_lbls.begin(), rhs2_lbls.end());

  internal::update_labels(labels);

  lhs_lbls  = IndexLabelVec(labels.begin(), labels.begin() + lhs_.labels().size());
  rhs1_lbls = IndexLabelVec(labels.begin() + lhs_.labels().size(),
                            labels.begin() + lhs_.labels().size() + rhs1_.labels().size());
  rhs2_lbls = IndexLabelVec(labels.begin() + lhs_.labels().size() + rhs1_.labels().size(),
                            labels.begin() + lhs_.labels().size() + rhs1_.labels().size() +
                              rhs2_.labels().size());
  lhs_.set_labels(lhs_lbls);
  rhs1_.set_labels(rhs1_lbls);
  rhs2_.set_labels(rhs2_lbls);

  // fillin_int_labels
  std::map<TileLabelElement, int> primary_labels_map;
  int                             cnt = -1;
  for(const auto& lbl: lhs_.labels()) { primary_labels_map[lbl.primary_label()] = --cnt; }
  for(const auto& lbl: rhs1_.labels()) { primary_labels_map[lbl.primary_label()] = --cnt; }
  for(const auto& lbl: rhs2_.labels()) { primary_labels_map[lbl.primary_label()] = --cnt; }
  for(const auto& lbl: lhs_.labels()) {
    lhs_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
  }
  for(const auto& lbl: rhs1_.labels()) {
    rhs1_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
  }
  for(const auto& lbl: rhs2_.labels()) {
    rhs2_int_labels_.push_back(primary_labels_map[lbl.primary_label()]);
  }
  // todo: validate

  using TensorElType1 = T;
  using TensorElType2 = T;
  using TensorElType3 = T;

  // determine set of all labels for do_work
  IndexLabelVec all_labels{lhs_.labels()};
  all_labels.insert(all_labels.end(), rhs1_.labels().begin(), rhs1_.labels().end());
  all_labels.insert(all_labels.end(), rhs2_.labels().begin(), rhs2_.labels().end());
  // LabelLoopNest loop_nest{all_labels};

  // execute-bufacc
  IndexLabelVec lhs_labels{lhs_.labels()};
  IndexLabelVec rhs1_labels{rhs1_.labels()};
  IndexLabelVec rhs2_labels{rhs2_.labels()};
  IndexLabelVec all_rhs_labels{rhs1_.labels()};
  all_rhs_labels.insert(all_rhs_labels.end(), rhs2_.labels().begin(), rhs2_.labels().end());

  // compute the reduction labels
  std::sort(lhs_labels.begin(), lhs_labels.end());
  auto unique_labels = internal::unique_entries_by_primary_label(all_rhs_labels);
  std::sort(unique_labels.begin(), unique_labels.end());
  IndexLabelVec reduction_labels; //{reduction.begin(), reduction.end()};
  std::set_difference(unique_labels.begin(), unique_labels.end(), lhs_labels.begin(),
                      lhs_labels.end(), std::back_inserter(reduction_labels));

  std::vector<int> rhs1_map_output;
  std::vector<int> rhs2_map_output;
  std::vector<int> rhs1_map_reduction;
  std::vector<int> rhs2_map_reduction;
  // const auto&      lhs_lbls = lhs_.labels();
  for(auto& lbl: rhs1_labels) {
    auto it_out = std::find(lhs_lbls.begin(), lhs_lbls.end(), lbl);
    if(it_out != lhs_lbls.end()) rhs1_map_output.push_back(it_out - lhs_lbls.begin());
    else rhs1_map_output.push_back(-1);

    // auto it_red = std::find(reduction.begin(), reduction.end(), lbl);
    auto it_red = std::find(reduction_labels.begin(), reduction_labels.end(), lbl);
    if(it_red != reduction_labels.end())
      rhs1_map_reduction.push_back(it_red - reduction_labels.begin());
    else rhs1_map_reduction.push_back(-1);
  }

  for(auto& lbl: rhs2_labels) {
    auto it_out = std::find(lhs_lbls.begin(), lhs_lbls.end(), lbl);
    if(it_out != lhs_lbls.end()) rhs2_map_output.push_back(it_out - lhs_lbls.begin());
    else rhs2_map_output.push_back(-1);

    auto it_red = std::find(reduction_labels.begin(), reduction_labels.end(), lbl);
    if(it_red != reduction_labels.end())
      rhs2_map_reduction.push_back(it_red - reduction_labels.begin());
    else rhs2_map_reduction.push_back(-1);
  }

  auto ctensor = lhs_.tensor();
  auto atensor = rhs1_.tensor();
  auto btensor = rhs2_.tensor();
  // for(auto itval=loop_nest.begin(); itval!=loop_nest.end(); ++itval) {}

  auto& oprof = tamm::OpProfiler::instance();

  auto compute_v4_term = [=, &oprof](const IndexVector& cblkid, span<T> cbuf) {
    auto& memHostPool = tamm::RMMMemoryManager::getInstance().getHostMemoryPool();

    // compute blockids from the loop indices. itval is the loop index
    // execute_bufacc(ec, hw);
    LabelLoopNest lhs_loop_nest{lhs_.labels()};
    IndexVector   translated_ablockid, translated_bblockid, translated_cblockid;
    auto          it    = lhs_loop_nest.begin();
    auto          itval = *it;
    for(; it != lhs_loop_nest.end(); ++it) {
      itval = *it;
      // auto        it   = ivec.begin();
      IndexVector c_block_id{itval};
      translated_cblockid = internal::translate_blockid(c_block_id, lhs_);
      if(translated_cblockid == cblkid) break;
    }

    // execute
    // const auto& ldist = lhs_.tensor().distribution();
    // for(const auto& lblockid: lhs_loop_nest) {
    //   const auto translated_lblockid = internal::translate_blockid(lblockid, lhs_);
    //   if(lhs_.tensor().is_non_zero(translated_lblockid) &&
    //       std::get<0>(ldist.locate(translated_lblockid)) == rank) {
    //     lambda(lblockid);
    //   }

    const size_t csize = ctensor.block_size(translated_cblockid);
    std::memset(cbuf.data(), 0, csize * sizeof(TensorElType1));
    const auto& cdims = ctensor.block_dims(translated_cblockid);

    SizeVec cdims_sz;
    for(const auto v: cdims) { cdims_sz.push_back(v); }

    AddBuf<TensorElType1, TensorElType2, TensorElType3>* ab{nullptr};
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
    TensorElType2* th_a{nullptr};
    TensorElType3* th_b{nullptr};
    auto&          thandle = GPUStreamPool::getInstance().getStream();

    ab =
      new AddBuf<TensorElType1, TensorElType2, TensorElType3>{th_a, th_b, {}, translated_cblockid};
#else
    gpuStream_t thandle{};
    ab = new AddBuf<TensorElType1, TensorElType2, TensorElType3>{ctensor, {}, translated_cblockid};
#endif

    // LabelLoopNest inner_loop{reduction_lbls};
    LabelLoopNest inner_loop{reduction_labels};

    // int loop_counter = 0;

    TensorElType1* cbuf_dev_ptr{nullptr};
    TensorElType1* cbuf_tmp_dev_ptr{nullptr};
#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
    auto& memDevicePool = tamm::RMMMemoryManager::getInstance().getDeviceMemoryPool();

    if(hw == ExecutionHW::GPU) {
      cbuf_dev_ptr =
        static_cast<TensorElType1*>(memDevicePool.allocate(csize * sizeof(TensorElType1)));
      cbuf_tmp_dev_ptr =
        static_cast<TensorElType1*>(memDevicePool.allocate(csize * sizeof(TensorElType1)));

      gpuMemsetAsync(reinterpret_cast<void*&>(cbuf_dev_ptr), csize * sizeof(TensorElType1),
                     thandle);
      gpuMemsetAsync(reinterpret_cast<void*&>(cbuf_tmp_dev_ptr), csize * sizeof(TensorElType1),
                     thandle);
    }
#endif

    for(const auto& inner_it_val: inner_loop) { // k

      IndexVector a_block_id(rhs1_.labels().size());

      for(size_t i = 0; i < rhs1_map_output.size(); i++) {
        if(rhs1_map_output[i] != -1) { a_block_id[i] = itval[rhs1_map_output[i]]; }
      }

      for(size_t i = 0; i < rhs1_map_reduction.size(); i++) {
        if(rhs1_map_reduction[i] != -1) { a_block_id[i] = inner_it_val[rhs1_map_reduction[i]]; }
      }

      const auto translated_ablockid = internal::translate_blockid(a_block_id, rhs1_);
      if(!atensor.is_non_zero(translated_ablockid)) continue;

      IndexVector b_block_id(rhs2_.labels().size());

      for(size_t i = 0; i < rhs2_map_output.size(); i++) {
        if(rhs2_map_output[i] != -1) { b_block_id[i] = itval[rhs2_map_output[i]]; }
      }

      for(size_t i = 0; i < rhs2_map_reduction.size(); i++) {
        if(rhs2_map_reduction[i] != -1) { b_block_id[i] = inner_it_val[rhs2_map_reduction[i]]; }
      }

      const auto translated_bblockid = internal::translate_blockid(b_block_id, rhs2_);
      if(!btensor.is_non_zero(translated_bblockid)) continue;

      // compute block size and allocate buffers for abuf and bbuf
      const size_t asize = atensor.block_size(translated_ablockid);
      const size_t bsize = btensor.block_size(translated_bblockid);

      TensorElType2* abuf{nullptr};
      TensorElType3* bbuf{nullptr};
      abuf = static_cast<TensorElType2*>(memHostPool.allocate(asize * sizeof(TensorElType2)));
      bbuf = static_cast<TensorElType3*>(memHostPool.allocate(bsize * sizeof(TensorElType3)));

      atensor.get(translated_ablockid, {abuf, asize});
      btensor.get(translated_bblockid, {bbuf, bsize});

      const auto& adims = atensor.block_dims(translated_ablockid);
      const auto& bdims = btensor.block_dims(translated_bblockid);

      // changed cscale from 0 to 1 to aggregate on cbuf
      T cscale{1};

      SizeVec adims_sz, bdims_sz;
      for(const auto v: adims) { adims_sz.push_back(v); }
      for(const auto v: bdims) { bdims_sz.push_back(v); }

      // A*B
      {
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
        TensorElType2* abuf_dev{nullptr};
        TensorElType3* bbuf_dev{nullptr};
        if(hw == ExecutionHW::GPU) {
          abuf_dev =
            static_cast<TensorElType2*>(memDevicePool.allocate(asize * sizeof(TensorElType2)));
          bbuf_dev =
            static_cast<TensorElType3*>(memDevicePool.allocate(bsize * sizeof(TensorElType3)));
          TimerGuard tg_copy{&oprof.multOpCopyTime};
          gpuMemcpyAsync<TensorElType2>(abuf_dev, abuf, asize, gpuMemcpyHostToDevice, thandle);
          gpuMemcpyAsync<TensorElType3>(bbuf_dev, bbuf, bsize, gpuMemcpyHostToDevice, thandle);
        }
#endif
        {
          TimerGuard tg_dgemm{&oprof.multOpDgemmTime};
          kernels::block_multiply<T, TensorElType1, TensorElType2, TensorElType3>(
#if defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP)
            abuf_dev, bbuf_dev,
#endif
            thandle, 1.0, abuf, adims_sz, rhs1_int_labels_, bbuf, bdims_sz, rhs2_int_labels_,
            cscale, cbuf.data(), cdims_sz, lhs_int_labels_, hw, false, cbuf_dev_ptr,
            cbuf_tmp_dev_ptr);
        }

#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
        if(hw == ExecutionHW::GPU) {
          memDevicePool.deallocate(abuf_dev, asize * sizeof(TensorElType2));
          memDevicePool.deallocate(bbuf_dev, bsize * sizeof(TensorElType3));
        }
#endif
      } // A * B

      memHostPool.deallocate(abuf, asize * sizeof(TensorElType2));
      memHostPool.deallocate(bbuf, bsize * sizeof(TensorElType3));
    } // end of reduction loop

    // add the computed update to the tensor
    {
#if(defined(USE_CUDA) || defined(USE_HIP) || defined(USE_DPCPP))
      // copy to host
      if(hw == ExecutionHW::GPU) {
        TensorElType1* cbuf_tmp{nullptr};
        cbuf_tmp = static_cast<TensorElType1*>(memHostPool.allocate(csize * sizeof(TensorElType1)));
        std::memset(cbuf_tmp, 0, csize * sizeof(TensorElType1));
        {
          TimerGuard tg_copy{&oprof.multOpCopyTime};
          gpuMemcpyAsync<TensorElType1>(cbuf_tmp, cbuf_dev_ptr, csize, gpuMemcpyDeviceToHost,
                                        thandle);
        }
        // cbuf+=cbuf_tmp
        gpuStreamSynchronize(thandle);
        blas::axpy(csize, TensorElType1{1}, cbuf_tmp, 1, cbuf.data(), 1);

        // free cbuf_dev_ptr
        memDevicePool.deallocate(static_cast<void*>(cbuf_dev_ptr), csize * sizeof(TensorElType1));
        memDevicePool.deallocate(static_cast<void*>(cbuf_tmp_dev_ptr),
                                 csize * sizeof(TensorElType1));

        memHostPool.deallocate(cbuf_tmp, csize * sizeof(TensorElType1));
      }
#endif
      // ctensor.add(translated_cblockid, cbuf);
      // for (size_t i=0;i<csize;i++) dbuf[i] = cbuf[i];
    }

    delete ab;
  };

  a22_abab = Tensor<T>{{v_alpha, v_beta, v_alpha, v_beta}, compute_v4_term};

  // clang-format off
  sch
    (_a017("aa")(p1_va, h2_oa, cind)         = -1.0  * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * chol3d_ov("aa")(h1_oa, p2_va, cind),
    "_a017( aa )(p1_va, h2_oa, cind)         = -1.0  * t2_aaaa_temp(p1_va, p2_va, h2_oa, h1_oa) * chol3d_ov( aa )(h1_oa, p2_va, cind)")
    (_a006("aa")(h2_oa, h1_oa)               = -1.0  * chol3d_ov("aa")(h2_oa, p2_va, cind) * _a017("aa")(p2_va, h1_oa, cind),
    "_a006( aa )(h2_oa, h1_oa)               = -1.0  * chol3d_ov( aa )(h2_oa, p2_va, cind) * _a017( aa )(p2_va, h1_oa, cind)")
    (_a007V(cind)                            =  2.0  * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p1_va, h1_oa),
    "_a007V(cind)                            =  2.0  * chol3d_ov( aa )(h1_oa, p1_va, cind) * t1_aa(p1_va, h1_oa)")
    (_a009("aa")(h1_oa, h2_oa, cind)         =  1.0  * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p1_va, h2_oa),
    "_a009( aa )(h1_oa, h2_oa, cind)         =  1.0  * chol3d_ov( aa )(h1_oa, p1_va, cind) * t1_aa(p1_va, h2_oa)")
    (_a021("aa")(p2_va, p1_va, cind)         = -0.5  * chol3d_ov("aa")(h1_oa, p1_va, cind) * t1_aa(p2_va, h1_oa),
    "_a021( aa )(p2_va, p1_va, cind)         = -0.5  * chol3d_ov( aa )(h1_oa, p1_va, cind) * t1_aa(p2_va, h1_oa)")
    (_a021("aa")(p2_va, p1_va, cind)        +=  0.5  * chol3d_vv("aa")(p2_va, p1_va, cind),
    "_a021( aa )(p2_va, p1_va, cind)        +=  0.5  * chol3d_vv( aa )(p2_va, p1_va, cind)")
    (_a017("aa")(p1_va, h2_oa, cind)        += -2.0  * t1_aa(p2_va, h2_oa) * _a021("aa")(p1_va, p2_va, cind),
    "_a017( aa )(p1_va, h2_oa, cind)        += -2.0  * t1_aa(p2_va, h2_oa) * _a021( aa )(p1_va, p2_va, cind)")
    (_a008("aa")(h2_oa, h1_oa, cind)         =  1.0  * _a009("aa")(h2_oa, h1_oa, cind),
    "_a008( aa )(h2_oa, h1_oa, cind)         =  1.0  * _a009( aa )(h2_oa, h1_oa, cind)")
    (_a009("aa")(h2_oa, h1_oa, cind)        +=  1.0  * chol3d_oo("aa")(h2_oa, h1_oa, cind),
    "_a009( aa )(h2_oa, h1_oa, cind)        +=  1.0  * chol3d_oo( aa )(h2_oa, h1_oa, cind)")
    .exact_copy(_a009("bb")(h2_ob,h1_ob,cind),_a009("aa")(h2_ob,h1_ob,cind))
    .exact_copy(_a021("bb")(p2_vb,p1_vb,cind),_a021("aa")(p2_vb,p1_vb,cind))
    (_a001("aa")(p1_va, p2_va)               = -2.0  * _a021("aa")(p1_va, p2_va, cind) * _a007V(cind),
    "_a001( aa )(p1_va, p2_va)               = -2.0  * _a021( aa )(p1_va, p2_va, cind) * _a007V(cind)")
    (_a001("aa")(p1_va, p2_va)              += -1.0  * _a017("aa")(p1_va, h2_oa, cind) * chol3d_ov("aa")(h2_oa, p2_va, cind),
    "_a001( aa )(p1_va, p2_va)              += -1.0  * _a017( aa )(p1_va, h2_oa, cind) * chol3d_ov( aa )(h2_oa, p2_va, cind)")
    (_a006("aa")(h2_oa, h1_oa)              +=  1.0  * _a009("aa")(h2_oa, h1_oa, cind) * _a007V(cind),
    "_a006( aa )(h2_oa, h1_oa)              +=  1.0  * _a009( aa )(h2_oa, h1_oa, cind) * _a007V(cind)")
    (_a006("aa")(h3_oa, h1_oa)              += -1.0  * _a009("aa")(h2_oa, h1_oa, cind) * _a008("aa")(h3_oa, h2_oa, cind),
    "_a006( aa )(h3_oa, h1_oa)              += -1.0  * _a009( aa )(h2_oa, h1_oa, cind) * _a008( aa )(h3_oa, h2_oa, cind)")
    (_a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob)  =  0.25 * _a009("aa")(h2_oa, h1_oa, cind) * _a009("bb")(h1_ob, h2_ob, cind),
    "_a019( abab )(h2_oa, h1_ob, h1_oa, h2_ob)  =  0.25 * _a009( aa )(h2_oa, h1_oa, cind) * _a009( bb )(h1_ob, h2_ob, cind)")
    (_a020("aaaa")(p2_va, h2_oa, p1_va, h1_oa)  = -2.0  * _a009("aa")(h2_oa, h1_oa, cind) * _a021("aa")(p2_va, p1_va, cind),
    "_a020( aaaa )(p2_va, h2_oa, p1_va, h1_oa)  = -2.0  * _a009( aa )(h2_oa, h1_oa, cind) * _a021( aa )(p2_va, p1_va, cind)")
    .exact_copy(_a020("baba")(p2_vb, h2_oa, p1_vb, h1_oa),_a020("aaaa")(p2_vb, h2_oa, p1_vb, h1_oa))
    (_a020("aaaa")(p1_va, h3_oa, p3_va, h2_oa) +=  0.5  * _a004("aaaa")(p2_va, p3_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa),
    "_a020( aaaa )(p1_va, h3_oa, p3_va, h2_oa) +=  0.5  * _a004( aaaa )(p2_va, p3_va, h3_oa, h1_oa) * t2_aaaa(p1_va,p2_va,h1_oa,h2_oa)")
    (_a020("baab")(p1_vb, h2_oa, p1_va, h2_ob)  = -0.5  * _a004("aaaa")(p2_va, p1_va, h2_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob),
    "_a020( baab )(p1_vb, h2_oa, p1_va, h2_ob)  = -0.5  * _a004( aaaa )(p2_va, p1_va, h2_oa, h1_oa) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)")
    (_a020("baba")(p1_vb, h1_oa, p2_vb, h2_oa) +=  0.5  * _a004("abab")(p1_va, p2_vb, h1_oa, h1_ob) * t2_abab(p1_va,p1_vb,h2_oa,h1_ob),
    "_a020( baba )(p1_vb, h1_oa, p2_vb, h2_oa) +=  0.5  * _a004( abab )(p1_va, p2_vb, h1_oa, h1_ob) * t2_abab(p1_va,p1_vb,h2_oa,h1_ob)")
    (_a017("aa")(p1_va, h2_oa, cind)           +=  1.0  * t1_aa(p1_va, h1_oa) * chol3d_oo("aa")(h1_oa, h2_oa, cind),
    "_a017( aa )(p1_va, h2_oa, cind)           +=  1.0  * t1_aa(p1_va, h1_oa) * chol3d_oo( aa )(h1_oa, h2_oa, cind)")
    (_a017("aa")(p1_va, h2_oa, cind)           += -1.0  * chol3d_ov("aa")(h2_oa, p1_va, cind),
    "_a017( aa )(p1_va, h2_oa, cind)           += -1.0  * chol3d_ov( aa )(h2_oa, p1_va, cind)")
    (_a001("aa")(p2_va, p1_va)                 += -1.0  * f1_vv("aa")(p2_va, p1_va),
    "_a001( aa )(p2_va, p1_va)                 += -1.0  * f1_vv( aa )(p2_va, p1_va)")
    (_a001("aa")(p2_va, p1_va)                 +=  1.0  * t1_aa(p2_va, h1_oa) * f1_ov("aa")(h1_oa, p1_va),
    "_a001( aa )(p2_va, p1_va)                 +=  1.0  * t1_aa(p2_va, h1_oa) * f1_ov( aa )(h1_oa, p1_va)") // NEW TERM
    (_a006("aa")(h2_oa, h1_oa)                 +=  1.0  * f1_oo("aa")(h2_oa, h1_oa),
    "_a006( aa )(h2_oa, h1_oa)                 +=  1.0  * f1_oo( aa )(h2_oa, h1_oa)")
    (_a006("aa")(h2_oa, h1_oa)                 +=  1.0  * t1_aa(p1_va, h1_oa) * f1_ov("aa")(h2_oa, p1_va),
    "_a006( aa )(h2_oa, h1_oa)                 +=  1.0  * t1_aa(p1_va, h1_oa) * f1_ov( aa )(h2_oa, p1_va)")
    .exact_copy(_a017("bb")(p1_vb, h1_ob, cind), _a017("aa")(p1_vb, h1_ob, cind))
    .exact_copy(_a006("bb")(h1_ob, h2_ob), _a006("aa")(h1_ob, h2_ob))
    .exact_copy(_a001("bb")(p1_vb, p2_vb), _a001("aa")(p1_vb, p2_vb))
    .exact_copy(_a021("bb")(p1_vb, p2_vb, cind), _a021("aa")(p1_vb, p2_vb, cind))
    .exact_copy(_a020("bbbb")(p1_vb, h1_ob, p2_vb, h2_ob), _a020("aaaa")(p1_vb, h1_ob, p2_vb, h2_ob))

    (i0_abab(p1_va, p2_vb, h2_oa, h1_ob)          =  1.0  * _a020("bbbb")(p2_vb, h2_ob, p1_vb, h1_ob) * t2_abab(p1_va, p1_vb, h2_oa, h2_ob),
    "i0_abab(p1_va, p2_vb, h2_oa, h1_ob)          =  1.0  * _a020(bbbb)(p2_vb, h2_ob, p1_vb, h1_ob) * t2_abab(p1_va, p1_vb, h2_oa, h2_ob)")
    (i0_abab(p2_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020("baab")(p1_vb, h1_oa, p1_va, h1_ob) * t2_aaaa(p2_va, p1_va, h2_oa, h1_oa),
    "i0_abab(p2_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020(baab)(p1_vb, h1_oa, p1_va, h1_ob) * t2_aaaa(p2_va, p1_va, h2_oa, h1_oa)")
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020("baba")(p1_vb, h1_oa, p2_vb, h2_oa) * t2_abab(p1_va, p2_vb, h1_oa, h1_ob),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * _a020(baba)(p1_vb, h1_oa, p2_vb, h2_oa) * t2_abab(p1_va, p2_vb, h1_oa, h1_ob)")
    .exact_copy(i0_temp(p1_vb,p1_va,h2_ob,h1_oa),i0_abab(p1_vb,p1_va,h2_ob,h1_oa))
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * i0_temp(p1_vb, p1_va, h1_ob, h2_oa),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         +=  1.0  * i0_temp(p1_vb, p1_va, h1_ob, h2_oa)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  1.0  * _a017("aa")(p1_va, h1_oa, cind) * _a017("bb")(p1_vb, h2_ob, cind),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  1.0  * _a017( aa )(p1_va, h1_oa, cind) * _a017( bb )(p1_vb, h2_ob, cind)");

    sch
    // (_a022("abab")(p1_va,p2_vb,p2_va,p1_vb)       =  1.0  * _a021("aa")(p1_va,p2_va,cind) * _a021("bb")(p2_vb,p1_vb,cind),
    // "_a022( abab )(p1_va,p2_vb,p2_va,p1_vb)       =  1.0  * _a021( aa )(p1_va,p2_va,cind) * _a021( bb )(p2_vb,p1_vb,cind)")
    (i0_abab(p1_va, p2_vb, h1_oa, h2_ob)         +=  4.0  * a22_abab(p1_va, p2_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob),
    "i0_abab(p1_va, p2_vb, h1_oa, h2_ob)         +=  4.0  * a22_abab(p1_va, p2_vb, p2_va, p1_vb) * t2_abab(p2_va,p1_vb,h1_oa,h2_ob)");


    sch(_a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob)   +=  0.25 * _a004("abab")(p1_va, p2_vb, h2_oa, h1_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob),
    "_a019( abab )(h2_oa, h1_ob, h1_oa, h2_ob)   +=  0.25 * _a004( abab )(p1_va, p2_vb, h2_oa, h1_ob) * t2_abab(p1_va,p2_vb,h1_oa,h2_ob)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  4.0  * _a019("abab")(h2_oa, h1_ob, h1_oa, h2_ob) * t2_abab(p1_va, p1_vb, h2_oa, h1_ob),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         +=  4.0  * _a019( abab )(h2_oa, h1_ob, h1_oa, h2_ob) * t2_abab(p1_va, p1_vb, h2_oa, h1_ob)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p2_vb, h1_oa, h2_ob) * _a001("bb")(p1_vb, p2_vb),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p2_vb, h1_oa, h2_ob) * _a001( bb )(p1_vb, p2_vb)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a001("aa")(p1_va, p2_va),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p2_va, p1_vb, h1_oa, h2_ob) * _a001( aa )(p1_va, p2_va)")
    (i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006("aa")(h1_oa, h2_oa),
    "i0_abab(p1_va, p1_vb, h2_oa, h1_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006( aa )(h1_oa, h2_oa)")
    (i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006("bb")(h1_ob, h2_ob),
    "i0_abab(p1_va, p1_vb, h1_oa, h2_ob)         += -1.0  * t2_abab(p1_va, p1_vb, h1_oa, h1_ob) * _a006( bb )(h1_ob, h2_ob)")
    ;
  // clang-format on
}

int main(int argc, char* argv[]) {
  using T = double;

  tamm::initialize(argc, argv);

  if(argc < 5) {
    tamm_terminate("Please provide occ_alpha, virt_alpha, cholesky-count and tile size");
  }

  size_t n_occ_alpha = atoi(argv[1]);
  size_t n_vir_alpha = atoi(argv[2]);
  size_t chol_count  = atoi(argv[3]);
  Tile   tile_size   = atoi(argv[4]);

  const auto nbf = n_occ_alpha + n_vir_alpha;

  ProcGroup        pg = ProcGroup::create_world_coll();
  ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};

  ExecutionHW exhw = ec.exhw();

  Scheduler sch{ec};

  bool profile = false;

  if(ec.print()) {
    std::cout << tamm_git_info() << std::endl;
    auto current_time   = std::chrono::system_clock::now();
    auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
    auto cur_local_time = localtime(&current_time_t);
    std::cout << std::endl << "date: " << std::put_time(cur_local_time, "%c") << std::endl;
    std::cout << "nnodes: " << ec.nnodes() << ", ";
    std::cout << "nproc_per_node: " << ec.ppn() << ", ";
    std::cout << "nproc_total: " << ec.nnodes() * ec.ppn() << ", ";
    std::cout << "ngpus_per_node: " << ec.gpn() << ", ";
    std::cout << "ngpus_total: " << ec.nnodes() * ec.gpn() << std::endl;
    ec.print_mem_info();
    std::cout << std::endl;
    std::cout << "basis functions: " << nbf << ", occ_alpha: " << n_occ_alpha
              << ", virt_alpha: " << n_vir_alpha << ", chol-count: " << chol_count
              << ", tilesize: " << tile_size << std::endl;
  }

  //-----------------------------------

  TAMM_SIZE n_occ_beta = n_occ_alpha;
  Tile      tce_tile   = tile_size;

  TAMM_SIZE nmo        = 2 * nbf;
  TAMM_SIZE n_vir_beta = n_vir_alpha;
  TAMM_SIZE nocc       = 2 * n_occ_alpha;

  const TAMM_SIZE total_orbitals = nmo;

  // Construction of tiled index space MO
  IndexSpace MO_IS{
    range(0, total_orbitals),
    {{"occ", {range(0, nocc)}},
     {"occ_alpha", {range(0, n_occ_alpha)}},
     {"occ_beta", {range(n_occ_alpha, nocc)}},
     {"virt", {range(nocc, total_orbitals)}},
     {"virt_alpha", {range(nocc, nocc + n_vir_alpha)}},
     {"virt_beta", {range(nocc + n_vir_alpha, total_orbitals)}}},
    {{Spin{1}, {range(0, n_occ_alpha), range(nocc, nocc + n_vir_alpha)}},
     {Spin{2}, {range(n_occ_alpha, nocc), range(nocc + n_vir_alpha, total_orbitals)}}}};

  std::vector<Tile> mo_tiles;

  tamm::Tile est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_occ_alpha / tce_tile));
  for(tamm::Tile x = 0; x < est_nt; x++)
    mo_tiles.push_back(n_occ_alpha / est_nt + (x < (n_occ_alpha % est_nt)));

  est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_occ_beta / tce_tile));
  for(tamm::Tile x = 0; x < est_nt; x++)
    mo_tiles.push_back(n_occ_beta / est_nt + (x < (n_occ_beta % est_nt)));

  est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_vir_alpha / tce_tile));
  for(tamm::Tile x = 0; x < est_nt; x++)
    mo_tiles.push_back(n_vir_alpha / est_nt + (x < (n_vir_alpha % est_nt)));

  est_nt = static_cast<tamm::Tile>(std::ceil(1.0 * n_vir_beta / tce_tile));
  for(tamm::Tile x = 0; x < est_nt; x++)
    mo_tiles.push_back(n_vir_beta / est_nt + (x < (n_vir_beta % est_nt)));

  TiledIndexSpace MO{MO_IS, mo_tiles};

  //----------------------------------------------------

  TiledIndexSpace N = MO("all");

  Tensor<T> d_f1{{N, N}, {1, 1}};
  Tensor<T>::allocate(&ec, d_f1);

  tamm::random_ip(d_f1);

  std::vector<T> p_evl_sorted;
  Tensor<T>      t1_aa, t2_abab, r1_aa, r2_abab;

  std::tie(p_evl_sorted, t1_aa, t2_abab, r1_aa, r2_abab) = setupTensors_cs(ec, MO, d_f1);

  IndexSpace      chol_is{range(0, chol_count)};
  TiledIndexSpace CI{chol_is, 1000};

  // cholVpr = {{N, N, CI}, {SpinPosition::upper, SpinPosition::lower, SpinPosition::ignore}};
  // Tensor<TensorType>::allocate(&ec, cholVpr);

  const TiledIndexSpace& O = MO("occ");
  const TiledIndexSpace& V = MO("virt");
  auto [cind]              = CI.labels<1>("all");

  const int otiles  = O.num_tiles();
  const int vtiles  = V.num_tiles();
  const int oatiles = MO("occ_alpha").num_tiles();
  const int vatiles = MO("virt_alpha").num_tiles();

  o_alpha = {MO("occ"), range(oatiles)};
  v_alpha = {MO("virt"), range(vatiles)};
  o_beta  = {MO("occ"), range(oatiles, otiles)};
  v_beta  = {MO("virt"), range(vatiles, vtiles)};

  auto [p1_va, p2_va] = v_alpha.labels<2>("all");
  auto [p1_vb, p2_vb] = v_beta.labels<2>("all");
  auto [h3_oa, h4_oa] = o_alpha.labels<2>("all");
  auto [h3_ob, h4_ob] = o_beta.labels<2>("all");

  Tensor<T> d_e{};

  Tensor<T> t2_aaaa = {{v_alpha, v_alpha, o_alpha, o_alpha}, {2, 2}};

  CCSE_Tensors<T> f1_oo{MO, {O, O}, "f1_oo", {"aa"}}; // bb
  CCSE_Tensors<T> f1_ov{MO, {O, V}, "f1_ov", {"aa"}}; // bb
  CCSE_Tensors<T> f1_vv{MO, {V, V}, "f1_vv", {"aa"}}; // bb

  CCSE_Tensors<T> chol3d_oo{MO, {O, O, CI}, "chol3d_oo", {"aa"}}; // bb
  CCSE_Tensors<T> chol3d_ov{MO, {O, V, CI}, "chol3d_ov", {"aa"}}; // bb
  CCSE_Tensors<T> chol3d_vv{MO, {V, V, CI}, "chol3d_vv", {"aa"}}; // bb

  std::vector<CCSE_Tensors<T>> f1_se{f1_oo, f1_ov, f1_vv};
  std::vector<CCSE_Tensors<T>> chol3d_se{chol3d_oo, chol3d_ov, chol3d_vv};

  _a01V = {CI};
  _a02  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a02", {"aa"}};
  _a03  = CCSE_Tensors<T>{MO, {O, V, CI}, "_a03", {"aa"}};
  _a004 = CCSE_Tensors<T>{MO, {V, V, O, O}, "_a004", {"aaaa", "abab"}};

  t2_aaaa_temp = {v_alpha, v_alpha, o_alpha, o_alpha};
  i0_temp      = {v_beta, v_alpha, o_beta, o_alpha};

  // Intermediates
  // T1
  _a02V = {CI};
  _a01  = CCSE_Tensors<T>{MO, {O, O, CI}, "_a01", {"aa"}};
  _a04  = CCSE_Tensors<T>{MO, {O, O}, "_a04", {"aa"}};
  _a05  = CCSE_Tensors<T>{MO, {O, V}, "_a05", {"aa"}}; // bb
  _a06  = CCSE_Tensors<T>{MO, {V, O, CI}, "_a06", {"aa"}};

  // T2
  _a007V = {CI};
  _a001  = CCSE_Tensors<T>{MO, {V, V}, "_a001", {"aa", "bb"}};
  _a006  = CCSE_Tensors<T>{MO, {O, O}, "_a006", {"aa", "bb"}};

  _a008 = CCSE_Tensors<T>{MO, {O, O, CI}, "_a008", {"aa"}};
  _a009 = CCSE_Tensors<T>{MO, {O, O, CI}, "_a009", {"aa", "bb"}};
  _a017 = CCSE_Tensors<T>{MO, {V, O, CI}, "_a017", {"aa", "bb"}};
  _a021 = CCSE_Tensors<T>{MO, {V, V, CI}, "_a021", {"aa", "bb"}};

  _a019 = CCSE_Tensors<T>{MO, {O, O, O, O}, "_a019", {"abab"}};
  // _a022 = CCSE_Tensors<T>{MO, {V, V, V, V}, "_a022", {"abab"}};
  _a020 = CCSE_Tensors<T>{MO, {V, O, V, O}, "_a020", {"aaaa", "baba", "baab", "bbbb"}};

  sch.allocate(t2_aaaa);
  sch.allocate(d_e, i0_temp, t2_aaaa_temp, _a01V);
  CCSE_Tensors<T>::allocate_list(sch, f1_oo, f1_ov, f1_vv, chol3d_oo, chol3d_ov, chol3d_vv);
  CCSE_Tensors<T>::allocate_list(sch, _a02, _a03);

  // allocate all intermediates
  sch.allocate(_a02V, _a007V);
  CCSE_Tensors<T>::allocate_list(sch, _a004, _a01, _a04, _a05, _a06, _a001, _a006, _a008, _a009,
                                 _a017, _a019, _a020, _a021);
  sch.execute();

  tamm::random_ip(f1_oo("aa"));
  tamm::random_ip(f1_ov("aa"));
  tamm::random_ip(f1_vv("aa"));
  tamm::random_ip(chol3d_oo("aa"));
  tamm::random_ip(chol3d_ov("aa"));
  tamm::random_ip(chol3d_vv("aa"));

  // clang-format off
  sch
      (_a004("aaaa")(p1_va, p2_va, h4_oa, h3_oa) = 1.0 * chol3d_ov("aa")(h4_oa, p1_va, cind) *
      chol3d_ov("aa")(h3_oa, p2_va, cind)) .exact_copy(_a004("abab")(p1_va, p1_vb, h3_oa, h3_ob),
      _a004("aaaa")(p1_va, p1_vb, h3_oa, h3_ob))
      ;
  // clang-format on

  sch.execute(exhw);

  const auto timer_start = std::chrono::high_resolution_clock::now();

  ccsd_e_cs(sch, MO, CI, d_e, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se);
  ccsd_t1_cs(sch, MO, CI, r1_aa, t1_aa, t2_abab, f1_se, chol3d_se);
  ccsd_t2_cs(sch, MO, CI, r2_abab, t1_aa, t2_abab, t2_aaaa, f1_se, chol3d_se);

  sch.execute(exhw, profile);

  const auto timer_end = std::chrono::high_resolution_clock::now();
  auto       iter_time =
    std::chrono::duration_cast<std::chrono::duration<double>>((timer_end - timer_start)).count();

  if(ec.print()) std::cout << "Time taken for closed-shell CD-CCSD: " << iter_time << std::endl;

  if(profile && ec.print()) {
    std::string profile_csv = "ccsd_profile_" + std::to_string(nbf) + "bf_" +
                              std::to_string(n_occ_alpha) + "oa_" + std::to_string(n_vir_alpha) +
                              "va_" + std::to_string(chol_count) + "cv_" +
                              std::to_string(tile_size) + "TS.csv";
    std::ofstream pds(profile_csv, std::ios::out);
    if(!pds) std::cerr << "Error opening file " << profile_csv << std::endl;
    pds << ec.get_profile_header() << std::endl;
    pds << ec.get_profile_data().str() << std::endl;
    pds.close();
  }

  if(false) {
    ExecutionContext ec_dense{ec.pg(), DistributionKind::dense, MemoryManagerKind::ga};
    Tensor<T>        c3dvv_dense = tamm::to_dense_tensor(ec_dense, chol3d_vv("aa"));
    print_dense_tensor(c3dvv_dense, "c3d_vv_aa_dense");
  }

  // deallocate all intermediates
  sch.deallocate(_a02V, _a007V);
  CCSE_Tensors<T>::deallocate_list(sch, _a004, _a01, _a04, _a05, _a06, _a001, _a006, _a008, _a009,
                                   _a017, _a019, _a020, _a021);

  sch.deallocate(d_e, i0_temp, t2_aaaa_temp, _a01V);
  CCSE_Tensors<T>::deallocate_list(sch, _a02, _a03);
  CCSE_Tensors<T>::deallocate_list(sch, f1_oo, f1_ov, f1_vv, chol3d_oo, chol3d_ov, chol3d_vv);

  sch.execute();

  sch.deallocate(t1_aa, t2_abab, r1_aa, r2_abab, d_f1, t2_aaaa).execute();

  tamm::finalize();

  return 0;
}
