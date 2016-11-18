//------------------------------------------------------------------------------
// Copyright (C) 2016, Pacific Northwest National Laboratory
// This software is subject to copyright protection under the laws of the
// United States and other countries
//
// All rights in this computer software are reserved by the
// Pacific Northwest National Laboratory (PNNL)
// Operated by Battelle for the U.S. Department of Energy
//
//------------------------------------------------------------------------------
#include "tensor/tensors_and_ops.h"
#include <memory>
#include <string>
#include <vector>
#include "tensor/equations.h"
#include "tensor/input.h"

using std::string;
using std::vector;

namespace tamm {

static Assignment consAddOp(const Equations &eqs, const IndexName *indices,
                            std::map<std::string, tamm::Tensor> *tensors,
                            const AddOp &add);

static Multiplication consMultOp(const Equations &eqs, const IndexName *indices,
                                 std::map<std::string, tamm::Tensor> *tensors,
                                 const MultOp &mult);

static Range2Index range2indices[] = {
    {12,
     {H1B, H2B, H3B, H4B, H5B, H6B, H7B, H8B, H9B, H10B, H11B, H12B}},  // TO
    {12,
     {P1B, P2B, P3B, P4B, P5B, P6B, P7B, P8B, P9B, P10B, P11B, P12B}},  // TV
    {0, {}}                                                             // TN
};

void tensors_and_ops(Equations *eqs,
                     std::map<std::string, tamm::Tensor> * tensors,
                     std::vector<Operation> * ops) {
  int inames[RANGE_UB] = {0};
  std::unique_ptr<RangeType[]> rts(new RangeType[eqs->range_entries.size()]);
  std::unique_ptr<IndexName[]> indices(new IndexName[
                                       eqs->index_entries.size()]);

  for (int i = 0; i < eqs->range_entries.size(); i++) {
    const char *rname = eqs->range_entries[i].name.c_str();
    if (!strcmp(rname, OSTR)) {
      rts[i] = TO;
      continue;
    } else if (!strcmp(rname, VSTR)) {
      rts[i] = TV;
      continue;
    } else if (!strcmp(rname, NSTR)) {
      rts[i] = TN;
      continue;
    } else {
      printf("Unsupported range type %s\n", rname);
      exit(1);
    }
  }

  for (int i = 0; i < eqs->index_entries.size(); i++) {
    int rid = eqs->index_entries[i].range_id;
    RangeType rt = rts[rid];
    assert(rt >= 0 && rt < RANGE_UB);
    assert(inames[rt] < range2indices[rt].nindices);
    indices[i] = range2indices[rt].names[inames[rt]++];
  }

  // tensors.resize(eqs.tensor_entries.size());
  // for(int i=0; i<eqs.tensor_entries.size(); i++) {
  tensors->clear();
  for (std::map<std::string, tamm::TensorEntry>::iterator i =
           eqs->tensor_entries.begin();
       i != eqs->tensor_entries.end(); i++) {
    RangeType ranges[MAX_TENSOR_DIMS];
    int ndim = i->second.ndim;
    for (int j = 0; j < ndim; j++) {
      ranges[j] = rts[i->second.range_ids[j]];
    }
    /*@BUG: @FIXME: dist_nw is a placeholder. Should be correct before this
     * object is used*/
    /*@BUG: @FIXME: irrep is not set.. Should be correctly set before this
     * object is used*/
    DistType bug_dist = dist_nw;
    int bug_irrep = 0;
    tensors->insert(std::map<std::string, tamm::Tensor>::value_type(
        string(i->first),
        Tensor(i->second.ndim, i->second.nupper, bug_irrep, ranges, bug_dist)));
  }

  // distributon, irrep
  ops->resize(eqs->op_entries.size());
  for (int i = 0; i < eqs->op_entries.size(); i++) {
    (*ops)[i].optype = eqs->op_entries[i].optype;
    switch (eqs->op_entries[i].optype) {
      case OpTypeAdd:
        (*ops)[i].add = consAddOp(*eqs, indices.get(), tensors,
                                  eqs->op_entries[i].add);
        break;
      case OpTypeMult:
        (*ops)[i].mult =
            consMultOp(*eqs, indices.get(), tensors, eqs->op_entries[i].mult);
        break;
      default:
        assert(0);
    }
  }
}

static Assignment consAddOp(const Equations &eqs, const IndexName *indices,
                            std::map<std::string, tamm::Tensor> *tensors,
                            const AddOp &add) {
  vector<IndexName> aids, cids;
  // assert(add);
  assert(eqs.tensor_entries.at(add.tc).ndim ==
         eqs.tensor_entries.at(add.ta).ndim);
  int ndim = eqs.tensor_entries.at(add.tc).ndim;

  assert(ndim > 0);
  aids.resize(ndim);
  cids.resize(ndim);
  for (int i = 0; i < ndim; i++) {
    aids[i] = indices[add.ta_ids[i]];
    cids[i] = indices[add.tc_ids[i]];
  }
  return Assignment(&(*tensors)[add.tc], &(*tensors)[add.ta], add.alpha, cids,
                    aids);
}

static Multiplication consMultOp(const Equations &eqs, const IndexName *indices,
                                 std::map<std::string, tamm::Tensor> *tensors,
                                 const MultOp mult) {
  vector<IndexName> aids, bids, cids;

  int cndim = eqs.tensor_entries.at(mult.tc).ndim;
  int andim = eqs.tensor_entries.at(mult.ta).ndim;
  int bndim = eqs.tensor_entries.at(mult.tb).ndim;
  assert(andim + bndim >= cndim);

  aids.resize(andim);
  bids.resize(bndim);
  cids.resize(cndim);

  for (int i = 0; i < andim; i++) {
    aids[i] = indices[mult.ta_ids[i]];
  }
  for (int i = 0; i < bndim; i++) {
    bids[i] = indices[mult.tb_ids[i]];
  }
  for (int i = 0; i < cndim; i++) {
    cids[i] = indices[mult.tc_ids[i]];
  }
  return Multiplication(&(*tensors)[mult.tc], cids, &(*tensors)[mult.ta], aids,
                        &(*tensors)[mult.tb], bids, mult.alpha);
}
}  // namespace tamm
